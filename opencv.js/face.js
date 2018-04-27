let videoHeight = 480
let videoWidth = 640
let outputCanvas = document.getElementById("outputCanvas");

let cap = null
let faceCascade = null;
let src = null;
let gray = null;

function run() {

    faceCascade = new cv.CascadeClassifier();
    faceCascade.load("face.xml")

    cap = new cv.VideoCapture(video)
    src = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
    gray = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

    startCamera();
    requestAnimationFrame(detectFace)
}

async function startCamera() {
    let video = document.getElementById("video");
    let stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: {
                exact: videoWidth
            },
            height: {
                exact: videoHeight
            }
        },
        audio: false
    })
    video.srcObject = stream;
    video.play();
}

function detectFace() {
    // Capture a frame
    cap.read(src)

    // Convert to greyscale
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);


    // Downsample
    let downSampled = new cv.Mat();
    cv.pyrDown(gray, downSampled);
    cv.pyrDown(downSampled, downSampled);

    // Detect faces
    let faces = new cv.RectVector();
    faceCascade.detectMultiScale(downSampled, faces)

    // Draw boxes
    let size = downSampled.size();
    let xRatio = videoWidth / size.width;
    let yRatio = videoHeight / size.height;
    for (let i = 0; i < faces.size(); ++i) {
        let face = faces.get(i);
        let point1 = new cv.Point(face.x * xRatio, face.y * yRatio);
        let point2 = new cv.Point((face.x + face.width) * xRatio, (face.y + face.height) * xRatio);
        cv.rectangle(src, point1, point2, [255, 0, 0, 255])
    }
    // Free memory
    downSampled.delete()
    faces.delete()

    // Show image
    cv.imshow(outputCanvas, src)

    requestAnimationFrame(detectFace)
}

// Config OpenCV
var Module = {
    locateFile: function (name) {
        let files = {
            "opencv_js.wasm": '/opencv/opencv_js.wasm'
        }
        return files[name]
    },
    preRun: [() => {
        Module.FS_createPreloadedFile("/", "face.xml", "model/haarcascade_frontalface_default.xml",
            true, false);
    }],
    postRun: [
        run
    ]
};