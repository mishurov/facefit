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

#include "prnet.h"

#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session_options.h>

using namespace tensorflow;

typedef std::vector<std::pair<std::string, Tensor>> tensor_dict;


PRNet::PRNet(const std::string& metaGraphPath,
		const std::string& checkpointPath)
{
	std::cout << "Loading the neural network...\n";
	Status status;
	SessionOptions options;
	TF_CHECK_OK(NewSession(options, &_sess));
	TF_CHECK_OK(loadModel(metaGraphPath, checkpointPath));
}


Tensor PRNet::infer(const tensorflow::Tensor& img)
{
	// std::cout << "Starting forward propagation...\n";
	tensor_dict feed_dict = {{"Placeholder", img}};
	std::vector<Tensor> outputs;
	Status status = _sess->Run(
		feed_dict,
		{"resfcn256/Conv2d_transpose_16/Sigmoid"},
		{},
		&outputs
	);
	Tensor output = outputs.at(0);
	return output;
}

// https://github.com/PatWie/tensorflow-cmake/
Status PRNet::loadModel(const std::string& metaGraphPath,
			const std::string& checkpointPath)
{
	Status status;

	// Read in the protobuf graph we exported
	MetaGraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), metaGraphPath, &graph_def);
	if (status != Status::OK())
		return status;

	// create the graph in the current session
	status = _sess->Create(graph_def.graph_def());
	if (status != Status::OK())
		return status;

	// restore model from checkpoint, iff checkpoint is given
	if (checkpointPath != "") {
		const std::string restore_op_name = graph_def.saver_def().restore_op_name();
		const std::string filename_tensor_name =
			graph_def.saver_def().filename_tensor_name();

		Tensor filename_tensor(DT_STRING, TensorShape());
		filename_tensor.scalar<std::string>()() = checkpointPath;

		tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};

		status = _sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
		if (status != Status::OK())
			return status;
	} else {
		status = _sess->Run({}, {}, {"init"}, nullptr);
		if (status != Status::OK())
			return status;
	}

	return Status::OK();
}
