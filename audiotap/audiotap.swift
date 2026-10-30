// audiotap — 2B's audio capture: the MIC and the Mac's system-audio OUTPUT, streamed as
// timestamped records on stdout, following devices as they come and go.
//
// Mic: an IOProc straight on the input device. `--mic auto` records the default input,
// except a Bluetooth headset (AirPods): then the built-in mic, which hears the whole room,
// doesn't drop out, and doesn't knock the headset into call-quality audio. `--mic default`
// always records the default input. The choice is re-checked every second: a mic that
// disconnects, stalls or changes sample rate is replaced, and a new default is followed.
//
// System audio: a Core Audio process tap (macOS 14.2+) on everything playing, to any
// device, inside a private aggregate device. No virtual driver, no change to the output
// device. The tap only delivers while some process is playing, so silence arrives as a
// gap in timestamps, not as zeros. It is rebuilt when the default output changes, and
// when something is playing but nothing arrives.
//
// Permissions belong to the app that launched us (Python, for the menubar): Microphone,
// and Screen & System Audio Recording > System Audio Recording Only. Without the latter
// the tap delivers silence rather than an error, so `audiotap permission` reports both.
//
// Usage:
//   audiotap record [--mic auto|default] [--no-system-audio] [--exclude-bundle-id ID]...
//                        stream until SIGTERM/SIGINT, the parent exits, or stdout closes
//   audiotap permission  JSON {"microphone": s, "system_audio": s}, s = granted|denied|unknown
//   audiotap inputs      JSON list of the input devices
//
// Records on stdout, little-endian:
//   char[4] source        "mic " | "sys "
//   uint64  host_ns       host time of the first sample (the CLOCK_UPTIME_RAW clock)
//   uint32  frames
//   float64 sample_rate
//   float32 samples[frames], mono
// Events on stderr, one JSON object per line: {"event": e, "host_ns": n, ...}
//   mic {device, transport, rate, reason}   recording from this mic from now on
//   no_mic {reason}                          no usable mic; waiting for one
//   output {device, transport}               the default output device (at start and on change)
//   tap {reason}                             the system-audio tap was (re)built
//   error {message}                          followed by exit 1
//
// Fail-hard: an unexpected Core Audio error emits an error event and exits 1. A device
// vanishing mid-query is expected and just means "gone".

import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation
import os

// MARK: - output

let writer = DispatchQueue(label: "audiotap.writer")

func nowNanos() -> UInt64 { AudioConvertHostTimeToNanos(AudioGetCurrentHostTime()) }

func emit(_ event: String, _ fields: [String: Any] = [:]) {
    var object = fields
    object["event"] = event
    object["host_ns"] = nowNanos()
    var line = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    line.append(0x0A)
    FileHandle.standardError.write(line)
}

func fail(_ message: String) -> Never {
    emit("error", ["message": message])
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail("\(what) failed: OSStatus \(status)") }
}

func writeAll(_ data: Data) {
    data.withUnsafeBytes { raw in
        var offset = 0
        while offset < raw.count {
            let n = write(STDOUT_FILENO, raw.baseAddress! + offset, raw.count - offset)
            if n < 0 {
                if errno == EINTR { continue }
                exit(0)  // the reader went away (EPIPE): nothing left to deliver to
            }
            offset += n
        }
    }
}

/// Queue one record for stdout. Called on the real-time IO threads, so never block here.
func send(_ source: String, hostNanos: UInt64, rate: Double, samples: [Float]) {
    var record = Data(capacity: 24 + 4 * samples.count)
    record.append(contentsOf: Array(source.utf8))
    withUnsafeBytes(of: hostNanos.littleEndian) { record.append(contentsOf: $0) }
    withUnsafeBytes(of: UInt32(samples.count).littleEndian) { record.append(contentsOf: $0) }
    withUnsafeBytes(of: rate.bitPattern.littleEndian) { record.append(contentsOf: $0) }
    samples.withUnsafeBytes { record.append(contentsOf: $0) }
    writer.async { writeAll(record) }
}

/// Average every channel of every buffer into one mono channel (Float32, as the HAL
/// presents it to IOProcs; checked when a device is opened).
func mixdown(_ list: UnsafePointer<AudioBufferList>) -> [Float] {
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
    guard let first = buffers.first, first.mNumberChannels > 0 else { return [] }
    let frames = Int(first.mDataByteSize) / (4 * Int(first.mNumberChannels))
    var mono = [Float](repeating: 0, count: frames)
    var channels = 0
    for buffer in buffers {
        guard let data = buffer.mData else { continue }
        let count = Int(buffer.mNumberChannels)
        let samples = data.assumingMemoryBound(to: Float.self)
        for c in 0..<count {
            for i in 0..<frames { mono[i] += samples[i * count + c] }
        }
        channels += count
    }
    if channels > 1 {
        let scale = 1 / Float(channels)
        for i in 0..<frames { mono[i] *= scale }
    }
    return mono
}

func hostNanos(_ time: UnsafePointer<AudioTimeStamp>) -> UInt64 {
    let stamp = time.pointee
    return AudioConvertHostTimeToNanos(stamp.mFlags.contains(.hostTimeValid) ? stamp.mHostTime : AudioGetCurrentHostTime())
}

// MARK: - Core Audio properties (nil = the object is gone or lacks the property)

func address(_ selector: AudioObjectPropertySelector,
             _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func property<T: BitwiseCopyable>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T,
                                  scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> T? {
    var addr = address(selector, scope)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    return AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr ? value : nil
}

func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr, let value else { return nil }
    return value.takeRetainedValue() as String
}

func objectList(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> [AudioObjectID] {
    var addr = address(selector, scope)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &addr, 0, nil, &size) == noErr else { return [] }
    var ids = [AudioObjectID](repeating: kAudioObjectUnknown, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

func fourCC(_ code: UInt32) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
    return String(bytes: bytes, encoding: .ascii) ?? String(code)
}

let system = AudioObjectID(kAudioObjectSystemObject)

struct Device: Equatable {
    let id: AudioObjectID
    let name: String
    let transport: String

    init?(_ id: AudioObjectID) {
        guard id != kAudioObjectUnknown,
              let name = stringProperty(id, kAudioObjectPropertyName),
              let transport = property(id, kAudioDevicePropertyTransportType, UInt32(0)) else { return nil }
        self.id = id
        self.name = name
        self.transport = fourCC(transport)
    }

    static func defaultDevice(_ selector: AudioObjectPropertySelector) -> Device? {
        property(system, selector, AudioObjectID(kAudioObjectUnknown)).flatMap(Device.init)
    }

    static func inputs() -> [Device] {
        objectList(system, kAudioHardwarePropertyDevices).filter {
            !objectList($0, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput).isEmpty
        }.compactMap(Device.init)
    }

    var isAlive: Bool { property(id, kAudioDevicePropertyDeviceIsAlive, UInt32(0)) == 1 }
    var nominalRate: Double? { property(id, kAudioDevicePropertyNominalSampleRate, Float64(0)) }
}

let bluetoothTransports: Set<String> = ["blue", "blea"]

// MARK: - mic

final class Mic {
    let mode: String
    private var device: Device?
    private var procID: AudioDeviceIOProcID?
    private var rate: Double = 0
    private var openedAt: UInt64 = 0
    private var waiting = false  // a no_mic event is out; don't repeat it every second
    private var stalls = 0  // consecutive reopens for silence; each doubles the patience
    private let lastIO = OSAllocatedUnfairLock(initialState: UInt64(0))

    init(mode: String) { self.mode = mode }

    func choose() -> Device? {
        let inputs = Device.inputs()
        let builtin = inputs.first { $0.transport == "bltn" }
        guard let current = Device.defaultDevice(kAudioHardwarePropertyDefaultInputDevice),
              inputs.contains(current) else { return builtin }
        if mode == "auto", bluetoothTransports.contains(current.transport), let builtin { return builtin }
        return current
    }

    /// Called every second: keep the right mic open and delivering.
    func reconcile() {
        let want = choose()
        let reason: String
        if let device {
            let last = lastIO.withLock { $0 }
            if last > openedAt { stalls = 0 }
            let silentFor = nowNanos() - max(last, openedAt)
            if !device.isAlive {
                reason = "\(device.name) disconnected"
            } else if device.nominalRate != rate {
                reason = "\(device.name) changed sample rate"
            } else if silentFor > 2_000_000_000 << UInt64(min(stalls, 5)) {
                stalls += 1
                reason = "\(device.name) stopped delivering audio"
            } else if want != device {
                reason = "input changed to \(want?.name ?? "nothing")"
            } else {
                return
            }
            close()
        } else if want != nil {
            reason = waiting ? "a mic appeared" : "start"
        } else {
            reason = "no input device"
        }
        guard let want else {
            if !waiting { emit("no_mic", ["reason": reason]) }
            waiting = true
            return
        }
        open(want, reason: reason)
    }

    private func open(_ candidate: Device, reason: String) {
        // The device can vanish at any point in here; then the next reconcile retries.
        guard let rate = candidate.nominalRate else { return }
        for stream in objectList(candidate.id, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput) {
            guard let format = property(stream, kAudioStreamPropertyVirtualFormat, AudioStreamBasicDescription()) else { return }
            if format.mFormatID != kAudioFormatLinearPCM || format.mFormatFlags & kAudioFormatFlagIsFloat == 0 || format.mBitsPerChannel != 32 {
                fail("mic \(candidate.name) delivers \(fourCC(format.mFormatID)) flags=\(format.mFormatFlags) bits=\(format.mBitsPerChannel), not Float32")
            }
        }
        var proc: AudioDeviceIOProcID?
        let lastIO = self.lastIO
        let status = AudioDeviceCreateIOProcIDWithBlock(&proc, candidate.id, nil) { _, input, inputTime, _, _ in
            let samples = mixdown(input)
            let ns = hostNanos(inputTime)
            lastIO.withLock { $0 = ns }
            if !samples.isEmpty { send("mic ", hostNanos: ns, rate: rate, samples: samples) }
        }
        guard status == noErr, let proc else { return }
        guard AudioDeviceStart(candidate.id, proc) == noErr else {
            AudioDeviceDestroyIOProcID(candidate.id, proc)
            return
        }
        device = candidate
        procID = proc
        self.rate = rate
        openedAt = nowNanos()
        waiting = false
        emit("mic", ["device": candidate.name, "transport": candidate.transport, "rate": rate, "reason": reason])
    }

    func close() {
        guard let device, let procID else { return }
        // Statuses ignored on purpose: the usual reason to close is that the device is gone.
        AudioDeviceStop(device.id, procID)
        AudioDeviceDestroyIOProcID(device.id, procID)
        self.device = nil
        self.procID = nil
    }
}

// MARK: - system audio

final class SystemTap {
    let excludeBundleIDs: Set<String>
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var deviceID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var output: Device?
    private var builtAt: UInt64 = 0
    private var stalls = 0  // consecutive rebuilds for silence; each doubles the patience
    private let lastIO = OSAllocatedUnfairLock(initialState: UInt64(0))

    init(excludeBundleIDs: Set<String>) { self.excludeBundleIDs = excludeBundleIDs }

    private func processes() -> [(id: AudioObjectID, bundleID: String)] {
        objectList(system, kAudioHardwarePropertyProcessObjectList).compactMap { process in
            stringProperty(process, kAudioProcessPropertyBundleID).map { (process, $0) }
        }
    }

    /// An app (other than the excluded ones) that is playing audio right now, if any.
    private func playingApp() -> String? {
        processes().first { process in
            !excludeBundleIDs.contains(process.bundleID)
                && property(process.id, kAudioProcessPropertyIsRunningOutput, UInt32(0)) == 1
        }?.bundleID
    }

    func build(reason: String) {
        let excluded = processes().filter { excludeBundleIDs.contains($0.bundleID) }.map(\.id)
        let description = CATapDescription(monoGlobalTapButExcludeProcesses: excluded)
        description.name = "2B system audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        check(AudioHardwareCreateProcessTap(description, &tapID), "AudioHardwareCreateProcessTap")

        guard let format = property(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()) else {
            fail("could not read the tap format")
        }
        guard format.mFormatID == kAudioFormatLinearPCM,
              format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              format.mBitsPerChannel == 32 else {
            fail("unexpected tap format: \(fourCC(format.mFormatID)) flags=\(format.mFormatFlags) bits=\(format.mBitsPerChannel)")
        }

        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "2B system audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [kAudioSubTapUIDKey: description.uuid.uuidString, kAudioSubTapDriftCompensationKey: true],
            ],
        ]
        check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &deviceID), "AudioHardwareCreateAggregateDevice")

        let rate = format.mSampleRate
        let lastIO = self.lastIO
        check(AudioDeviceCreateIOProcIDWithBlock(&procID, deviceID, nil) { _, input, inputTime, _, _ in
            let samples = mixdown(input)
            let ns = hostNanos(inputTime)
            lastIO.withLock { $0 = ns }
            if !samples.isEmpty { send("sys ", hostNanos: ns, rate: rate, samples: samples) }
        }, "AudioDeviceCreateIOProcIDWithBlock")
        check(AudioDeviceStart(deviceID, procID), "AudioDeviceStart")

        builtAt = nowNanos()
        emit("tap", ["reason": reason, "rate": rate])
        let current = Device.defaultDevice(kAudioHardwarePropertyDefaultOutputDevice)
        if current != output || reason == "start" {
            output = current
            emit("output", ["device": current?.name ?? "none", "transport": current?.transport ?? "none"])
        }
    }

    func destroy() {
        check(AudioDeviceStop(deviceID, procID), "AudioDeviceStop")
        check(AudioDeviceDestroyIOProcID(deviceID, procID!), "AudioDeviceDestroyIOProcID")
        check(AudioHardwareDestroyAggregateDevice(deviceID), "AudioHardwareDestroyAggregateDevice")
        check(AudioHardwareDestroyProcessTap(tapID), "AudioHardwareDestroyProcessTap")
        procID = nil
    }

    /// Called every second: rebuild the tap for a new output device, or when it has
    /// stopped delivering although an app is playing.
    func reconcile() {
        let current = Device.defaultDevice(kAudioHardwarePropertyDefaultOutputDevice)
        if current != output {
            destroy()
            build(reason: "output changed to \(current?.name ?? "nothing")")
            return
        }
        let last = lastIO.withLock { $0 }
        if last > builtAt { stalls = 0 }
        let silentFor = nowNanos() - max(last, builtAt)
        if silentFor > 2_000_000_000 << UInt64(min(stalls, 5)), let app = playingApp() {
            stalls += 1
            destroy()
            build(reason: "no audio arrived while \(app) was playing")
        }
    }
}

// MARK: - commands

func permission() {
    let microphone: String
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: microphone = "granted"
    case .denied, .restricted: microphone = "denied"
    default: microphone = "unknown"
    }
    // System Audio Recording has no public API: ask TCC directly. "unknown" = not asked
    // yet (starting a tap shows the prompt), or the SPI is unavailable.
    var systemAudio = "unknown"
    if let tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW),
       let symbol = dlsym(tcc, "TCCAccessPreflight") {
        typealias Preflight = @convention(c) (CFString, CFDictionary?) -> Int
        switch unsafeBitCast(symbol, to: Preflight.self)("kTCCServiceAudioCapture" as CFString, nil) {
        case 0: systemAudio = "granted"
        case 1: systemAudio = "denied"
        default: break
        }
    }
    let json = try! JSONSerialization.data(withJSONObject: ["microphone": microphone, "system_audio": systemAudio], options: [.sortedKeys])
    print(String(data: json, encoding: .utf8)!)
}

func inputs() {
    let current = Device.defaultDevice(kAudioHardwarePropertyDefaultInputDevice)
    let list = Device.inputs().map { ["name": $0.name, "transport": $0.transport, "default": $0 == current] as [String: Any] }
    let json = try! JSONSerialization.data(withJSONObject: list, options: [.sortedKeys])
    print(String(data: json, encoding: .utf8)!)
}

func record(mic mode: String, systemAudio: Bool, exclude: Set<String>) -> Never {
    signal(SIGPIPE, SIG_IGN)  // a closed stdout surfaces as EPIPE in writeAll instead

    let mic = Mic(mode: mode)
    let tap = systemAudio ? SystemTap(excludeBundleIDs: exclude) : nil
    mic.reconcile()
    tap?.build(reason: "start")

    func shutdown() -> Never {
        mic.close()
        tap?.destroy()
        writer.sync {}  // flush records already queued
        exit(0)
    }

    var sources: [DispatchSourceProtocol] = []
    let timer = DispatchSource.makeTimerSource(queue: .main)
    timer.schedule(deadline: .now() + 1, repeating: 1)
    timer.setEventHandler {
        mic.reconcile()
        tap?.reconcile()
    }
    timer.resume()
    sources.append(timer)
    for sig in [SIGTERM, SIGINT] {
        signal(sig, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        source.setEventHandler { shutdown() }
        source.resume()
        sources.append(source)
    }
    // The recorder (Python) owns us: if it dies, stop rather than record forever.
    let parent = DispatchSource.makeProcessSource(identifier: getppid(), eventMask: .exit, queue: .main)
    parent.setEventHandler { shutdown() }
    parent.resume()
    sources.append(parent)

    dispatchMain()
}

var args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "permission":
    permission()
case "inputs":
    inputs()
case "record":
    args.removeFirst()
    var mode = "auto"
    var systemAudio = true
    var exclude = Set<String>()
    while !args.isEmpty {
        switch (args.removeFirst(), args.first) {
        case ("--mic", let value?) where value == "auto" || value == "default":
            mode = args.removeFirst()
        case ("--exclude-bundle-id", let value?):
            exclude.insert(value)
            args.removeFirst()
        case ("--no-system-audio", _):
            systemAudio = false
        case (let flag, _):
            fail("unknown argument \(flag)")
        }
    }
    record(mic: mode, systemAudio: systemAudio, exclude: exclude)
default:
    fail("usage: audiotap record|permission|inputs")
}
