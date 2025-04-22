//  FaceRecognitionView.swift
//  FaceDetector
//  Detect faces in real-time and match with registered faces

import SwiftUI
import AVFoundation
import Vision
import CoreML
import UIKit

class FaceRecognitionViewModel: ObservableObject {
    @Published var recognizedName: String = ""
    @Published var capturedImage: UIImage?
    var controller: FaceRecognitionController?

    func recognize() {
        controller?.captureCurrentFrame()
    }
}

class FaceRecognitionController: UIViewController {
    private var session: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var facenetModel: facenet?
    private var output: AVCaptureVideoDataOutput!
    private var currentPixelBuffer: CVPixelBuffer?
    private var viewModel: FaceRecognitionViewModel
    private var registeredFaces: [RegisteredFace] = []

    init(viewModel: FaceRecognitionViewModel) {
        self.viewModel = viewModel
        super.init(nibName: nil, bundle: nil)
        self.viewModel.controller = self
        self.loadRegisteredFaces()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        loadModel()
        setupCamera()
    }

    private func loadModel() {
        facenetModel = try? facenet(configuration: MLModelConfiguration())
    }

    private func loadRegisteredFaces() {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("faces.json")
        guard let data = try? Data(contentsOf: url) else { return }
        registeredFaces = (try? JSONDecoder().decode([RegisteredFace].self, from: data)) ?? []
    }

    private func setupCamera() {
        session = AVCaptureSession()
        session.sessionPreset = .photo

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Unable to access front camera")
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }

    func captureCurrentFrame() {
        guard let pixelBuffer = currentPixelBuffer else { return }
        detectFace(in: pixelBuffer)
    }

    private func detectFace(in pixelBuffer: CVPixelBuffer) {
        let request = VNDetectFaceRectanglesRequest { [weak self] request, error in
            guard let self = self else { return }
            if let results = request.results as? [VNFaceObservation], let face = results.first {
                self.extractEmbedding(from: pixelBuffer, face: face)
            } else {
                DispatchQueue.main.async {
                    self.viewModel.recognizedName = "No Face"
                }
            }
        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

    private func extractEmbedding(from pixelBuffer: CVPixelBuffer, face: VNFaceObservation) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let size = ciImage.extent.size
        let faceRect = CGRect(
            x: face.boundingBox.origin.x * size.width,
            y: (1 - face.boundingBox.origin.y - face.boundingBox.size.height) * size.height,
            width: face.boundingBox.size.width * size.width,
            height: face.boundingBox.size.height * size.height
        )

        let cropped = ciImage.cropped(to: faceRect)
        let context = CIContext()
        guard let cgImage = context.createCGImage(cropped, from: cropped.extent) else { return }
        let mirrored = UIImage(cgImage: cgImage, scale: 1.0, orientation: .leftMirrored)
        let resized = mirrored.resize(to: CGSize(width: 160, height: 160))

        guard let inputBuffer = resized.toCVPixelBuffer(),
              let result = try? facenetModel?.prediction(input__0: inputBuffer) else {
            print("Failed to extract embedding")
            return
        }

        let embedding = (0..<result.output__0.count).map { Double(truncating: result.output__0[$0]) }
        let match = matchFace(to: embedding)

        DispatchQueue.main.async {
            self.viewModel.recognizedName = match ?? "Unknown"
            self.viewModel.capturedImage = resized
        }
    }

    private func matchFace(to embedding: [Double]) -> String? {
        var bestMatch: (name: String, distance: Double)? = nil
        for face in registeredFaces {
            let distance = cosineDistance(a: embedding, b: face.embedding)
            if bestMatch == nil || distance < bestMatch!.distance {
                bestMatch = (face.name, distance)
            }
        }
        return (bestMatch?.distance ?? 1.0) < 0.6 ? bestMatch?.name : nil
    }

    private func cosineDistance(a: [Double], b: [Double]) -> Double {
        let dot = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        return 1 - (dot / (normA * normB))
    }
}

extension FaceRecognitionController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        currentPixelBuffer = pixelBuffer
    }
}

struct FaceRecognitionPreview: UIViewControllerRepresentable {
    @ObservedObject var viewModel: FaceRecognitionViewModel

    func makeUIViewController(context: Context) -> FaceRecognitionController {
        return FaceRecognitionController(viewModel: viewModel)
    }

    func updateUIViewController(_ uiViewController: FaceRecognitionController, context: Context) {}
}

struct FaceRecognitionView: View {
    @StateObject var viewModel = FaceRecognitionViewModel()

    var body: some View {
        ZStack {
            FaceRecognitionPreview(viewModel: viewModel)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()

                if let image = viewModel.capturedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 120, height: 120)
                        .clipShape(Circle())
                        .padding()
                }

                Text("Recognized: \(viewModel.recognizedName)")
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding()

                Button("Scan Face") {
                    viewModel.recognize()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)

                Spacer().frame(height: 40)
            }
        }
    }
}

