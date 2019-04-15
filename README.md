# Nuke connector for PRNet

It is the [PRNet](https://github.com/YadiraF/PRNet) imported into the Nuke compositing universe in the form of a binary plug-in.

It creates a geometry of a face from a single image. I use C++ TensorFlow API for evaluating the model and Dlib for basic image processing for moving data back and forth between TensorFlow and Nuke.

### Notes on installation
Download the pre-trained model from [here](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing) or [here](https://drive.google.com/open?id=1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH). And put it into ```data/net-data/```.

Change paths to the data, if needed, in constants in ```src/facefit.h```, I suppose that the resulting binary looks for the files from Nuke's executable directory.

Change paths to Nuke's directory in ```CMakeLists```. As well some compilation options there and in ```dependencies.sh``` i.e. CUDA, BLAS and whatnot.

I was compiling the project with gcc 6.3.0 and got into linking troubles regarding the ```_GLIBCXX_USE_CXX11_ABI```. It's worth paying attention to that.

The file ```dependencies.sh``` downloads and compiles TensorFlow C++ libraries, builds a Python package - takes quite a bit of time - downloads Dlib and saves PRNet's meta graph for loading in C++.

Initially it requires some dependencies i.e. build-essential or so. If something goes wrong, you can analyse the script and errors.

As model is downloaded, dependencies are installed and metagraph is saved, you can run
```sh
mkdir build
cd build
cmake ..
make
```

I don't know how to package the result, in my development setting I'm just symlinking the resulting .so into a Nuke's plug-in directory, e.g.

```sh
ln -s FaceFit.so ~/.nuke/
```

The binary reads external files from the data directory and it uses Tensorflow's shared libraries since TensorFlow's Bazel build system still can't do static libraries and I have no idea of its current status with Windows.


### Notes on implementation
The code processes image and point data almost naively in nested for loops, I guess it can be optimised via data parallelism.

I don't use correctly Nuke's logging and simply print into stdout some debug information.

The plug-in actively uses CUDA, I suppose the same code compiled for CPU will be much slower which may lead to a not very pleasant experience.

Output animation is jumpy. Well, it is basically raw data for each frame computed independently. One can use some Euclidean metric for a low pass filter or so to make it more outlier-resistant etc.

[Video](https://vimeo.com/330519572)


[![](http://mishurov.co.uk/images/github/facefit/boris1.png)](https://vimeo.com/330519572)


[![](http://mishurov.co.uk/images/github/facefit/boris2.png)](https://vimeo.com/330519572)

