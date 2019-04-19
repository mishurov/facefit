/* ************************************************************************
 * Copyright 2019 Alexander Mishurov
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#include "tflite_mesh.h"

#include <thread>
#include <iostream>
#define LOG(x) std::cerr


TFLiteMesh::TFLiteMesh(const std::string& modelPath, int resolution, int outputSize)
{
	_outputSize = outputSize;
	_resolution = resolution;
	_inputSize = _resolution * _resolution * 3;

	_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
	if (!_model) {
		LOG(FATAL) << "Failed to mmap model " << modelPath << "\n";
		return;
	}
	LOG(INFO) << "Loaded model " << modelPath << "\n";

	tflite::InterpreterBuilder(*_model, _resolver)(&_interpreter);
	if (!_interpreter) {
		LOG(FATAL) << "Failed to construct interpreter\n";
		return;
	}

	_inputIdx = _interpreter->inputs()[0];
	auto input_dims = _interpreter->tensor(_inputIdx)->dims->data;
	if (input_dims[1] * input_dims[2] * input_dims[3] != _inputSize) {
		LOG(FATAL) << "Model's input size doesn't match the configured one\n";
		return;
	}
	_outputMeshIdx = _interpreter->outputs()[0];
	auto output_dims = _interpreter->tensor(_outputMeshIdx)->dims->data;
	if (output_dims[3] != _outputSize) {
		LOG(FATAL) << "Model's output size doesn't match the configured one\n";
		return;
	}
	_outputProbIdx = _interpreter->outputs()[1];

	_interpreter->SetAllowFp16PrecisionForFp32(true);

	unsigned int nthreads = std::thread::hardware_concurrency();
	if (nthreads != 0) {
		_interpreter->SetNumThreads(nthreads);
	}

	if (_interpreter->AllocateTensors() != kTfLiteOk) {
		LOG(FATAL) << "Failed to allocate tensors!";
		return;
	}
}


float TFLiteMesh::infer(float* inputData, float* outputData)
{
	float* input = _interpreter->typed_tensor<float>(_inputIdx);

	std::copy(inputData, inputData + _inputSize, input);

	if (_interpreter->Invoke() != kTfLiteOk) {
		LOG(FATAL) << "Failed to invoke tflite!\n";
	}

	float* output = _interpreter->typed_tensor<float>(_outputMeshIdx);
	std::copy(output, output + _outputSize, outputData);
	
	return _interpreter->typed_tensor<float>(_outputProbIdx)[0];
}
