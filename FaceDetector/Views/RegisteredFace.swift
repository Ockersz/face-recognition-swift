//  FaceCaptureView.swift
//  FaceDetector
//  Enhanced for commercial-grade UX, performance, and UI improvements with manual crop selection

import SwiftUI
import AVFoundation
import Vision
import CoreML
import UIKit

struct RegisteredFace: Codable {
    let name: String
    let embedding: [Double]
}

@MainActor
class FaceCaptureViewModel: ObservableObject {
    @Published var name: String = ""
    @Published var capturedImage: UIImage?
    @Published var showCaptureConfirmation = false
    @Published var errorMessage: String?
    @Published var showCropSheet = false
    var controller: FaceCaptureController?
    
    func capture() {
        controller?.captureCurrentFrame()
    }
}

class FaceCaptureController: UIViewController {
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var facenetModel: facenet?
    private var viewModel: FaceCaptureViewModel
    private var currentPixelBuffer: CVPixelBuffer?
    private var output: AVCaptureVideoDataOutput!
    private let sequenceHandler = VNSequenceRequestHandler()
    
    init(viewModel: FaceCaptureViewModel) {
        self.viewModel = viewModel
        super.init(nibName: nil, bundle: nil)
        self.viewModel.controller = self
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
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device) else {
            DispatchQueue.main.async {
                self.viewModel.errorMessage = "Unable to access front camera."
            }
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(previewLayer, at: 0)
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func captureCurrentFrame() {
        guard let pixelBuffer = currentPixelBuffer else {
            viewModel.errorMessage = "No camera frame available."
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: .leftMirrored)
        
        DispatchQueue.main.async {
            self.viewModel.capturedImage = image
            self.viewModel.showCropSheet = true
        }
    }
    
    func finalizeEmbedding(from image: UIImage) {
        let resized = image.resize(to: CGSize(width: 160, height: 160))
        
        guard let inputBuffer = resized.toCVPixelBuffer(),
              let result = try? facenetModel?.prediction(input__0: inputBuffer) else {
            DispatchQueue.main.async {
                self.viewModel.errorMessage = "Failed to extract embedding."
            }
            return
        }
        
        let embedding = result.output__0
        let doubleEmbedding = (0..<embedding.count).map { Double(truncating: embedding[$0]) }
        
        DispatchQueue.main.async {
            self.viewModel.capturedImage = resized
            self.saveEmbedding(doubleEmbedding)
            self.viewModel.showCaptureConfirmation = true
        }
    }

    
    private func saveEmbedding(_ embedding: [Double]) {
        let newFace = RegisteredFace(name: viewModel.name, embedding: embedding)
        var faces = loadAllFaces()
        faces.append(newFace)
        
        if let data = try? JSONEncoder().encode(faces) {
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("faces.json")
            try? data.write(to: url)
        }
    }
    
    func loadAllFaces() -> [RegisteredFace] {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("faces.json")
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([RegisteredFace].self, from: data)) ?? []
    }
}

extension FaceCaptureController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        currentPixelBuffer = pixelBuffer
    }
}

struct FaceCapturePreview: UIViewControllerRepresentable {
    @ObservedObject var viewModel: FaceCaptureViewModel
    
    func makeUIViewController(context: Context) -> FaceCaptureController {
        return FaceCaptureController(viewModel: viewModel)
    }
    
    func updateUIViewController(_ uiViewController: FaceCaptureController, context: Context) {}
}

struct FaceCaptureView: View {
    @StateObject var viewModel = FaceCaptureViewModel()
    @State private var cropRect = CGRect.zero
    
    var body: some View {
        ZStack {
            FaceCapturePreview(viewModel: viewModel)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                
                VStack(spacing: 12) {
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }
                    
                    TextField("Enter Name", text: $viewModel.name)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .padding(.horizontal)
                        .background(Color.white.opacity(0.8))
                        .cornerRadius(10)
                    
                    Button(action: {
                        viewModel.capture()
                    }) {
                        Text("Capture Face")
                            .bold()
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                }
                .padding()
                
                if let image = viewModel.capturedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 160, height: 160)
                        .cornerRadius(10)
                        .padding(.top)
                }
                
                if viewModel.showCaptureConfirmation {
                    Text("✅ Face registered successfully!")
                        .foregroundColor(.green)
                        .padding(.top, 8)
                }
            }
        }
        .sheet(isPresented: $viewModel.showCropSheet) {
            if let uiImage = viewModel.capturedImage {
                ManualCropView(image: uiImage) { cropped in
                    viewModel.controller?.finalizeEmbedding(from: cropped)
                }
            } else {
                Text("No image to crop")
            }
        }
    }
}

struct ManualCropView: View {
    let image: UIImage
    let onDone: (UIImage) -> Void
    @Environment(\.dismiss) var dismiss
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    
    var body: some View {
        VStack {
            Spacer()
            
            GeometryReader { geo in
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .scaleEffect(scale)
                    .offset(offset)
                    .gesture(
                        MagnificationGesture()
                            .onChanged { scale in
                                self.scale = scale
                            }
                    )
                    .gesture(
                        DragGesture()
                            .onChanged { gesture in
                                self.offset = gesture.translation
                            }
                    )
            }
            
            Spacer()
            
            Button("Done") {
                let renderer = UIGraphicsImageRenderer(size: CGSize(width: 160, height: 160))
                let cropped = renderer.image { _ in
                    image.draw(in: CGRect(x: 0, y: 0, width: 160, height: 160))
                }
                onDone(cropped)
                dismiss()
            }
            .padding()
        }
    }
}

extension UIImage {
    func resize(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        draw(in: CGRect(origin: .zero, size: size))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage ?? self
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                width: Int(size.width),
                                height: Int(size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        if let context = context, let cgImage = self.cgImage {
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        }
        
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}
