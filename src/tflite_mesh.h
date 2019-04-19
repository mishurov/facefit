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

#ifndef TFLITEMESH_H_
#define TFLITEMESH_H_

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <string>


class TFLiteMesh {
public:
	TFLiteMesh(const std::string& modelPath, int resolution, int numPoints);
	float infer(float* inputData, float* outputSize);
private:
	std::unique_ptr<tflite::FlatBufferModel> _model;
	std::unique_ptr<tflite::Interpreter> _interpreter;
	tflite::ops::builtin::BuiltinOpResolver _resolver;
	int _resolution;
	int _inputSize;
	int _outputSize;

	int _inputIdx;
	int _outputMeshIdx;
	int _outputProbIdx;
};

#endif // TFLITEMESH_H_
