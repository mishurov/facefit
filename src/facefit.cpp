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

#include "facefit.h"
#include <DDImage/Knobs.h>
#include <DDImage/Point.h>
#include <DDImage/PolyMesh.h>
#include <DDImage/Polygon.h>


using namespace DD::Image;
using namespace facefit;


Nuke2TensorFlow::StaticData Nuke2TensorFlow::data(kFaceObjPath);


FaceFitOp::FaceFitOp(Node* node) :
	SourceGeo(node),
	_faceDetector(true),
	_cf{1, 0, 0},
	_bBox{0, 0, 0, 0},
	_updateReqInc(0),
	_n2tf(kInputResolution),
	_net(kModelPath, kInputResolution, kOutPointsNum * 3)
{
	std::cout << "FaceFitOp constructor.\n";
}


const char* FaceFitOp::input_label(int input, char* buffer) const
{
	if (input == 1)
		return "tex";
	return SourceGeo::input_label(input, buffer);
}


Iop* FaceFitOp::default_material_iop() const
{
	// SourceGeo's test_input() accepts only Iops as it is. Thus there's
	// no need in overriding test_input() and default_input(). Yet I don't
	// want iop_input() had been used as a material, only for bulding
	// geometry. I return a default Iop from SourceGeo so as to Nuke
	// weren't crashing due to accesses to a non-allocated memory.
	if (input1())
		return input1()->iop();
	return SourceGeo::default_input(0)->iop();
}


int FaceFitOp::minimum_inputs() const { return 1; }

int FaceFitOp::maximum_inputs() const { return 2; }


void FaceFitOp::knobs(Knob_Callback f)
{
	SourceGeo::knobs(f);
	Bool_knob(f, &_faceDetector,"detect_face", "detect face");
	BBox_knob(f, _bBox, "bounding_box", "face bounds");
	Color_knob(f, _cf, "colour", "colour");
	Button(f, "request_infer", "request infer");
}


int FaceFitOp::knob_changed(Knob* k)
{
	if (k == &Knob::showPanel) {
		knob("bounding_box")->enable(!_faceDetector);
		return 1;
	}

	if (k->is("detect_face"))  {
		knob("bounding_box")->enable(!_faceDetector);
		return 1;
	}

	if (k->is("request_infer"))  {
		std::cout << "Update requested.\n";
		_updateReqInc++;
		invalidateSameHash();
		return 1;
	}

	return SourceGeo::knob_changed(k);
}


void FaceFitOp::infer(bool modify)
{
	auto defaultPoints = Nuke2TensorFlow::data.defaultPoints();

	if (!modify) {
		_bufferPoints.resize(defaultPoints.size());
		std::copy(defaultPoints.begin(),
			defaultPoints.end(), _bufferPoints.begin());
	}

	if (input_iop() == default_input(0)->iop())
		return;

	Iop *inputIop = input_iop();

	Format format = inputIop->format();
	Box iopBox(0, 0, format.width(), format.height());

	Channel channelMask[3] = { Chan_Red, Chan_Green, Chan_Blue };
	auto channels = ChannelSet(channelMask, 3);

	input_iop()->request(iopBox, channels, 0);
	ImagePlane iopPlane = ImagePlane(iopBox, false, channels);
	input_iop()->fetchPlane(iopPlane);
	Box bBox(_bBox[0], _bBox[1], _bBox[2], _bBox[3]);

	float input[kInputResolution * kInputResolution * 3];
	if(!_n2tf.imagePlane2tflite(iopPlane, bBox, _faceDetector, input)) {
		std::cout << "error extracting facial data\n";
		return;
	}

	float output[kOutPointsNum * 3];
	float faceFlag = _net.infer(input, output);
	//std::cout << "Face flag " << faceFlag << "\n";

	_n2tf.extractDataFromTFLite(output);
	_bufferPoints.resize(defaultPoints.size());

	std::move(_n2tf.points().begin(), _n2tf.points().end(),
			_bufferPoints.begin());
}


void FaceFitOp::create_geometry(Scene& scene, GeometryList& out)
{
	int obj = 0;

	if (rebuild(Mask_Primitives)) {
		out.delete_objects();
		out.add_object(obj);

		auto indices = Nuke2TensorFlow::data.faceIndices();
		auto mesh = new PolyMesh(indices.size(), indices.size() / 3);
		for(int i = 2; i < indices.size(); i += 3) {
			int corners[3] =
				{ indices[i], indices[i - 1], indices[i - 2] };
			mesh->add_face(3, corners);
		}
		out.add_primitive(obj, mesh);
	}

	if (rebuild(Mask_Points)) {
		PointList* points = out.writable_points(obj);
		infer(points->size() > 0);
		points->resize(_bufferPoints.size());
		std::copy(_bufferPoints.begin(), _bufferPoints.end(),
				points->begin());
	}

	if (rebuild(Mask_Attributes)) {
		auto objInfo = out.object(obj);
		PointList* points = out.writable_points(obj);

		bool uvExists = false;
		for (int i = 0; i < objInfo.get_attribcontext_count(); i++) {
			auto context = objInfo.get_attribcontext(i);
			if (std::strcmp(context->name, "uv") == 0)
				uvExists = true;
			
		}

		if (!uvExists) {
			Attribute* uvs = out.writable_attribute(obj, Group_Points,
							"uv", VECTOR4_ATTRIB);
			assert(uvs);
			for (int i = 0; i < points->size(); i++) {
				uvs->vector4(i).set(
					Nuke2TensorFlow::data.uvs().at(i));
			}
		}
		Attribute* ca = out.writable_attribute(obj, Group_Points,
							"Cf", VECTOR4_ATTRIB);
		assert(ca);
		auto c = ca->vector4(1);
		if (c.x != _cf[0] || c.y != _cf[1] || c.z != _cf[2]) {
			for (int i = 0; i < points->size(); i++) {
				ca->vector4(i).set(
					_cf[0], _cf[1], _cf[2], 1.0f
				);
			}
		}
	}
}


void FaceFitOp::get_geometry_hash()
{
	SourceGeo::get_geometry_hash();

	geo_hash[Group_Points].append(input_iop()->hash());
	geo_hash[Group_Points].append(_updateReqInc);

	geo_hash[Group_Attributes].append(_cf[0]);
	geo_hash[Group_Attributes].append(_cf[1]);
	geo_hash[Group_Attributes].append(_cf[2]);
	geo_hash[Group_Attributes].append(default_material_iop()->hash());
}


Op* FaceFitOp::Build(Node* node) { return new FaceFitOp(node); }


const char* FaceFitOp::Class() const { return kFaceFitClass; }


const Op::Description FaceFitOp::description(kFaceFitClass, FaceFitOp::Build);
