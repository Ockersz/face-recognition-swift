//
//  ContentView.swift
//  FaceDetector
//
//  Created by Shahein Ockersz on 2025-04-19.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                NavigationLink("Register Face") {
                    FaceCaptureView()
                }
                .buttonStyle(.borderedProminent)
                
                
                NavigationLink("Recognize Faces") {
                     FaceRecognitionView()
                 }
                 .buttonStyle(.borderedProminent)
            }
            .navigationTitle("Face Detector")
        }
    }
}

