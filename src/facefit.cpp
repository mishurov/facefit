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


// Nuke can create several instances even for a single node,
// it's better to load NN models and related data into static variables.
Nuke2TensorFlow::StaticData Nuke2TensorFlow::data(
			kDetectorModelPath, kTrianglesPath, kFaceIndicesPath,
			kKptIndicesPath, kPRNetResolution);
// Without thread_local TF, being statically initialised, steals the UI thread.
// However, I suspect there're more elegant solutions for tackling this.
thread_local PRNet FaceFitOp::_net = PRNet(kMetaGraphPath, kCheckpointPath);


FaceFitOp::FaceFitOp(Node* node) :
	SourceGeo(node),
	_outType(0),
	_faceDetector(true),
	_cf{1, 0, 0},
	_bBox{0, 0, 0, 0},
	_updateReqInc(0),
	_pointRadius(5.0f),
	_n2tf(kPRNetResolution)
{
	std::cout << "FaceFitOp constructor.\n";
	_currentOutType = -1;
	_currentPointRadius = -1;
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
	Enumeration_knob(f, &_outType, _outTypeNames, "out_type", "out");
	Color_knob(f, _cf, "colour", "colour");
	Float_knob(f, &_pointRadius, "point_radius", "point radius");
	SetRange(f, 0.1, 4);
	Button(f, "request_infer", "request infer");
}


int FaceFitOp::knob_changed(Knob* k)
{
	if (k == &Knob::showPanel) {
		knob("bounding_box")->enable(!_faceDetector);
		knob("point_radius")->enable(_outType != kMesh);
		return 1;
	}

	if (k->is("detect_face"))  {
		knob("bounding_box")->enable(!_faceDetector);
		return 1;
	}

	if (k->is("out_type"))  {
		knob("point_radius")->enable(_outType != kMesh);
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

	tensorflow::Tensor input = _n2tf.imagePlane2Tensor(
					iopPlane, bBox, _faceDetector);
	if (input.dims() != 4) {
		std::cout << "Couldn't process input image.\n";
		return;
	}

	auto output = _net.infer(input);
	if (output.dims() != 4) {
		std::cout << "Couldn't process output tensor.\n";
		return;
	}
	
	//std::cout << "Extracting inferred data...\n";
	_n2tf.extractDataFromTensor(output);
	_bufferPoints.resize(defaultPoints.size());
	std::move(_n2tf.points().begin(), _n2tf.points().end(),
			_bufferPoints.begin());
}


void FaceFitOp::recreate_primitives(int obj, GeometryList& out,
				const std::vector<int>& indices) 
{
	out.delete_objects();
	out.add_object(obj);
	
	if (_outType == kMesh) {
		auto tris = Nuke2TensorFlow::data.triIndices();
		auto f2a = Nuke2TensorFlow::data.face2all();

		auto mesh = new PolyMesh(tris.size(), tris.size() / 3);
		for(int i = 2; i < tris.size(); i += 3) {
			int corners[3] = {
				f2a[tris[i]],
				f2a[tris[i - 1]],
				f2a[tris[i - 2]]
			};
			mesh->add_face(3, corners);
		}
		out.add_primitive(obj, mesh);

	} else {
		auto endList = Nuke2TensorFlow::data.endList();
		int start;
		for (int i = 0; i < indices.size(); i++) {
			int index = indices[i];

			out.add_primitive(obj,
				new Point(Point::RenderMode::DISC,
						_pointRadius, index)
			);

			if (_outType == kPointCloud)
				continue;

			if (endList.find(i) != endList.end()) {
				// shapes starting from 41 are closed
				if (i >= 41) {
					out.add_primitive(obj,
						new Polygon(index, start, false)
					);
				}
				start = indices[i + 1];
				continue;
			}

			out.add_primitive(obj,
				new Polygon(index, indices[i + 1], false)
			);
		}
		_currentPointRadius = _pointRadius;
	}
	_currentOutType = _outType;
}


void FaceFitOp::create_geometry(Scene& scene, GeometryList& out)
{
	auto indices = _outType != kKeyPoints ?
					Nuke2TensorFlow::data.faceIndices() :
					Nuke2TensorFlow::data.kptIndices();
	int obj = 0;

	if (rebuild(Mask_Primitives)) {
		recreate_primitives(obj, out, indices);
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

		if (_currentOutType != _outType) {
			// save points from current obj before deleting
			_bufferPoints.resize(
				Nuke2TensorFlow::data.defaultPoints().size()
			);
			std::move(points->begin(), points->end(),
					_bufferPoints.begin());

			recreate_primitives(obj, out, indices);
			// restore collected points
			points = out.writable_points(obj);
			points->resize(_bufferPoints.size());
			std::move(_bufferPoints.begin(), _bufferPoints.end(),
					points->begin());

			objInfo = out.object(obj);
		}

		if (_outType != kMesh && _currentPointRadius != _pointRadius) {
			for (int i = 0; i < objInfo.primitives(); i++) {
				auto prim = objInfo.primitive(i);
				// casts away constness, may be not legit
				if (std::strcmp(prim->Class(), "Point") == 0)
					((Point*)prim)->radius(_pointRadius);
			}
			_currentPointRadius = _pointRadius;
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

		bool uvExists = false;
		for (int i = 0; i < objInfo.get_attribcontext_count(); i++) {
			auto context = objInfo.get_attribcontext(i);
			if (std::strcmp(context->name, "uv") == 0)
				uvExists = true;
			
		}

		if (!uvExists) {
			Attribute* uva = out.writable_attribute(obj, Group_Points,
							"uv", VECTOR4_ATTRIB);
			assert(uva);
			for (int i = 0; i < points->size(); i++) {
				uva->vector4(i).set(
					Nuke2TensorFlow::data.uvs().at(i));
			}
		}
	}
}


void FaceFitOp::get_geometry_hash()
{
	SourceGeo::get_geometry_hash();

	// Mask_Primitives should be called only once during initialisation

	// Recompute point locations only if input image has changed
	geo_hash[Group_Points].append(input_iop()->hash());
	geo_hash[Group_Points].append(_updateReqInc);

	// I use Mask_Attributes instead of Mask_Points for recreting primitives
	// i.e. for geomoetry or facial points or key points because
	// it affects only visibiliy of primitives linked to the points.
	// Point_Mask is being used for changing positions after inference.
	// Since inference is rather slow, point locations should be recomputed
	// as few times as possible.
	geo_hash[Group_Attributes].append(_outType);

	geo_hash[Group_Attributes].append(_cf[0]);
	geo_hash[Group_Attributes].append(_cf[1]);
	geo_hash[Group_Attributes].append(_cf[2]);
	geo_hash[Group_Attributes].append(_pointRadius);
	
	// Redraw if texture changed
	geo_hash[Group_Points].append(default_material_iop()->hash());
}


Op* FaceFitOp::Build(Node* node) { return new FaceFitOp(node); }


const char* FaceFitOp::Class() const { return kFaceFitClass; }


const Op::Description FaceFitOp::description(kFaceFitClass, FaceFitOp::Build);
