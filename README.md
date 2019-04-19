# Nuke connector for ARCores's TensorFlow Lite model

A binary plug-in for Nuke.

It creates a geometry of a face from a single image. I use C++ TensorFlow API for evaluating the model and Dlib for basic image processing for moving data back and forth between TensorFlow and Nuke.

### Notes on installation
Change paths to the data, if needed, in constants in ```src/facefit.h```, I suppose that the resulting binary looks for the files from Nuke's executable directory.

Change paths to Nuke's directory in ```CMakeLists```. As well some compilation options there and in ```dependencies.sh```.

I was compiling the project with gcc 6.3.0 and got into linking troubles regarding the ```_GLIBCXX_USE_CXX11_ABI```. It's worth paying attention to that.

The file ```dependencies.sh``` downloads and compiles TensorFlow C++ libraries and downloads Dlib.

Initially it requires some dependencies i.e. build-essential or so. If something goes wrong, you can analyse the script and errors.

As dependencies are installed, you can run
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

### Notes on implementation
The code processes image and point data almost naively in nested for loops, I guess it can be optimised via data parallelism.

I don't use correctly Nuke's logging and simply print into stdout some debug information.

Output animation is jumpy. Well, it is basically raw data for each frame computed independently. One can use some Euclidean metric for a low pass filter or so to make it more outlier-resistant etc.

![](http://mishurov.co.uk/images/github/facefit/jacob.png "")
