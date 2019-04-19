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

#ifndef FACEFIT_H_
#define FACEFIT_H_

#include "nuke2tf.h"
#include "tflite_mesh.h"
#include <DDImage/Iop.h>
#include <DDImage/SourceGeo.h>


namespace facefit {

// path to the directory with model and vertex data,
// relative to Nuke's executable, I guess
static const std::string kDataPath = "../Documents/projects/facefit/data/";

// Paths to the actual files
static const std::string kFaceObjPath = kDataPath + "face_model_468.obj";
// there're two models acceptiong different resultions, both work
static const int kInputResolution = 192;
static const std::string kModelPath =
	kDataPath + "facemesh-lite_nocrawl-2019_01_14-v0.tflite";
//static const int kInputResolution = 128;
//static const std::string kModelPath =
//	kDataPath + "facemesh-ultralite_nocrawl-2018_12_21-v0.tflite";


static const char* kFaceFitClass = "FaceFit";
static const int kOutPointsNum = 468;


using namespace DD::Image;


class FaceFitOp : public SourceGeo {
public:
	FaceFitOp(Node* node);
	virtual const char* Class() const;
	virtual void knobs(Knob_Callback f);
	virtual int knob_changed(Knob* k);
	static Op* Build(Node* node);
	static const Description description;
	virtual const char* input_label(int input, char* buffer) const;
	virtual Iop* default_material_iop () const;
	int minimum_inputs() const;
	int maximum_inputs() const;
protected:
	virtual void create_geometry(Scene& scene,
					GeometryList& out);
	virtual void get_geometry_hash();

private:
	TFLiteMesh _net;
	Nuke2TensorFlow _n2tf;
	PointList _bufferPoints;

	// knobs
	bool _faceDetector;
	float _bBox[4];
	float _cf[3];
	unsigned _updateReqInc;

	void infer(bool modify);
	void recreate_primitives(int obj, GeometryList& out,
				const std::vector<int>& indices);

}; // class


}; // namespace
#endif // FACEFIT_H_
