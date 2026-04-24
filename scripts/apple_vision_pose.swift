import AVFoundation
import CoreGraphics
import Foundation
import Vision

struct Arguments {
    let videoPath: String
    let outputPath: String
    let confidence: Float
}

struct PoseFrame: Encodable {
    let frame: Int
    let time: Double
    let joints: [String: [Double]]
}

enum PoseError: Error, CustomStringConvertible {
    case usage(String)
    case asset(String)
    case reader(String)

    var description: String {
        switch self {
        case .usage(let message), .asset(let message), .reader(let message):
            return message
        }
    }
}

struct AppleVisionPoseExtractor {
    static func parseArguments() throws -> Arguments {
        var values: [String: String] = [:]
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = iterator.next() {
            guard arg.starts(with: "--"), let value = iterator.next() else {
                throw PoseError.usage("Usage: apple_vision_pose.swift --video <path> --out <path> [--confidence 0.2]")
            }
            values[String(arg.dropFirst(2))] = value
        }
        guard let videoPath = values["video"], let outputPath = values["out"] else {
            throw PoseError.usage("Usage: apple_vision_pose.swift --video <path> --out <path> [--confidence 0.2]")
        }
        let confidence = Float(values["confidence"] ?? "0.2") ?? 0.2
        return Arguments(videoPath: videoPath, outputPath: outputPath, confidence: confidence)
    }

    static func extractPoses(arguments: Arguments) throws {
        let videoURL = URL(fileURLWithPath: arguments.videoPath)
        let asset = AVURLAsset(url: videoURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw PoseError.asset("No video track found in \(arguments.videoPath)")
        }

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw PoseError.reader("Unable to attach video output reader")
        }
        reader.add(output)

        let request = VNDetectHumanBodyPoseRequest()
        let sequenceHandler = VNSequenceRequestHandler()

        let outURL = URL(fileURLWithPath: arguments.outputPath)
        try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: outURL)
        defer { try? handle.close() }

        guard reader.startReading() else {
            throw PoseError.reader("Failed to start video reader")
        }

        var frameIndex = 0
        while reader.status == .reading {
            autoreleasepool {
                guard let sampleBuffer = output.copyNextSampleBuffer() else { return }
                defer { CMSampleBufferInvalidate(sampleBuffer) }

                let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                let timeSeconds = CMTimeGetSeconds(timestamp)
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    frameIndex += 1
                    return
                }

                do {
                    try sequenceHandler.perform([request], on: pixelBuffer)
                    guard let observation = request.results?.first,
                          let frame = try poseFrame(from: observation, frameIndex: frameIndex, time: timeSeconds, confidenceThreshold: arguments.confidence) else {
                        frameIndex += 1
                        return
                    }
                    let data = try JSONEncoder().encode(frame)
                    handle.write(data)
                    handle.write("\n".data(using: .utf8)!)
                } catch {
                    // Skip failed frames but continue processing the sequence.
                }

                frameIndex += 1
            }
        }

        if reader.status == .failed {
            throw PoseError.reader(reader.error?.localizedDescription ?? "Video reader failed unexpectedly")
        }
    }

    static func poseFrame(
        from observation: VNHumanBodyPoseObservation,
        frameIndex: Int,
        time: Double,
        confidenceThreshold: Float
    ) throws -> PoseFrame? {
        let allPoints = try observation.recognizedPoints(.all)

        let jointPatterns: [String: String] = [
            "leftHip": "left_upLeg_joint",
            "rightHip": "right_upLeg_joint",
            "leftAnkle": "left_foot_joint",
            "rightAnkle": "right_foot_joint",
            "leftShoulder": "left_shoulder_1_joint",
            "rightShoulder": "right_shoulder_1_joint",
            "leftElbow": "left_forearm_joint",
            "rightElbow": "right_forearm_joint",
            "leftWrist": "left_hand_joint",
            "rightWrist": "right_hand_joint",
            "leftKnee": "left_leg_joint",
            "rightKnee": "right_leg_joint",
            "neck": "neck_1_joint",
            "head": "head_joint",
            "root": "root",
        ]

        func point(_ raw: String) -> CGPoint? {
            guard let pattern = jointPatterns[raw] else { return nil }
            guard let (_, p) = allPoints.first(where: { String(describing: $0.key).contains(pattern) }),
                  p.confidence >= confidenceThreshold else {
                return nil
            }
            return CGPoint(x: p.location.x, y: p.location.y)
        }

        guard let leftHip2D = point("leftHip"),
              let rightHip2D = point("rightHip"),
              let leftAnkle2D = point("leftAnkle"),
              let rightAnkle2D = point("rightAnkle"),
              let leftShoulder2D = point("leftShoulder"),
              let rightShoulder2D = point("rightShoulder") else {
            return nil
        }

        let hipCenter = midpoint(leftHip2D, rightHip2D)
        let shoulderCenter = midpoint(leftShoulder2D, rightShoulder2D)
        let ankleCenterY = (leftAnkle2D.y + rightAnkle2D.y) * 0.5
        let bodyHeight = max(ankleCenterY - shoulderCenter.y, 0.15)
        let scale = 1.42 / bodyHeight

        func convert(_ p: CGPoint) -> [Double] {
            [
                Double((hipCenter.x - p.x) * scale),
                Double((p.y - hipCenter.y) * scale + 0.95),
                0.0,
            ]
        }

        var joints: [String: [Double]] = [:]
        let names: [(String, String)] = [
            ("left_shoulder", "leftShoulder"),
            ("right_shoulder", "rightShoulder"),
            ("left_elbow", "leftElbow"),
            ("right_elbow", "rightElbow"),
            ("left_wrist", "leftWrist"),
            ("right_wrist", "rightWrist"),
            ("left_hip", "leftHip"),
            ("right_hip", "rightHip"),
            ("left_knee", "leftKnee"),
            ("right_knee", "rightKnee"),
            ("left_ankle", "leftAnkle"),
            ("right_ankle", "rightAnkle"),
            ("neck", "neck"),
            ("head", "head"),
            ("root", "root"),
        ]

        for (target, visionName) in names {
            if let p = point(visionName) {
                joints[target] = convert(p)
            }
        }

        guard let leftShoulder = joints["left_shoulder"],
              let rightShoulder = joints["right_shoulder"],
              let leftElbow = joints["left_elbow"],
              let rightElbow = joints["right_elbow"],
              let leftWrist = joints["left_wrist"],
              let rightWrist = joints["right_wrist"],
              let leftHip = joints["left_hip"],
              let rightHip = joints["right_hip"] else {
            return nil
        }

        let pelvis = joints["root"] ?? midpoint(leftHip, rightHip)
        let shoulderMid = midpoint(leftShoulder, rightShoulder)
        joints["pelvis"] = pelvis
        joints["spine"] = midpoint(pelvis, shoulderMid)

        if let neck = joints["neck"] {
            joints["neck"] = neck
        } else {
            joints["neck"] = weightedMidpoint(shoulderMid, joints["head"] ?? shoulderMid, 0.38)
        }

        if joints["head"] == nil {
            joints["head"] = [
                joints["neck"]?[0] ?? 0.0,
                (joints["neck"]?[1] ?? 1.5) + 0.16,
                0.0,
            ]
        }
        joints.removeValue(forKey: "root")

        let required = [leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist]
        guard required.count == 6 else { return nil }

        return PoseFrame(frame: frameIndex, time: time, joints: joints)
    }

    static func midpoint(_ a: CGPoint, _ b: CGPoint) -> CGPoint {
        CGPoint(x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5)
    }

    static func midpoint(_ a: [Double], _ b: [Double]) -> [Double] {
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]
    }

    static func weightedMidpoint(_ a: [Double], _ b: [Double], _ bWeight: Double) -> [Double] {
        let aWeight = 1.0 - bWeight
        return [
            a[0] * aWeight + b[0] * bWeight,
            a[1] * aWeight + b[1] * bWeight,
            a[2] * aWeight + b[2] * bWeight,
        ]
    }
}

do {
    let args = try AppleVisionPoseExtractor.parseArguments()
    try AppleVisionPoseExtractor.extractPoses(arguments: args)
} catch let error as PoseError {
    fputs("error: \(error.description)\n", stderr)
    Foundation.exit(1)
} catch {
    fputs("error: \(error)\n", stderr)
    Foundation.exit(1)
}
