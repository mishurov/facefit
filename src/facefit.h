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
#include "prnet.h"
#include <DDImage/Iop.h>
#include <DDImage/SourceGeo.h>


namespace facefit {

// path to the directory with model and vertex data,
// relative to Nuke's executable, I guess
static const std::string kDataPath = "../Documents/projects/facefit/data";

// TensorFlow model name
static const std::string kModelName = "256_256_resfcn256_weight";
// Paths to the actual files
static const std::string kMetaGraphPath =
				kDataPath + "/net-data/" + kModelName + ".meta";
static const std::string kCheckpointPath =
				kDataPath + "/net-data/" + kModelName;
static const std::string kDetectorModelPath =
			kDataPath + "/net-data/mmod_human_face_detector.dat";
static const std::string kFaceIndicesPath =
				kDataPath + "/uv-data/face_ind.txt";
static const std::string kKptIndicesPath =
				kDataPath + "/uv-data/uv_kpt_ind.txt";
static const std::string kTrianglesPath =
				kDataPath + "/uv-data/triangles.txt";

static const char* kFaceFitClass = "FaceFit";
static const int kPRNetResolution = 256;


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
	static thread_local PRNet _net;
	Nuke2TensorFlow _n2tf;
	PointList _bufferPoints;

	// knobs
	bool _pointCloud;
	bool _faceDetector;
	float _bBox[4];
	float _pointRadius;
	float _cf[3];
	const char* const _outTypeNames[4] =
			{ "key points", "mesh", "point cloud", 0 };
	enum _outTypes { kKeyPoints, kMesh, kPointCloud };
	int _outType;
	unsigned _updateReqInc;

	int _currentOutType;
	float _currentPointRadius;

	void infer(bool modify);
	void recreate_primitives(int obj, GeometryList& out,
				const std::vector<int>& indices);

}; // class


}; // namespace
#endif // FACEFIT_H_
