"""Functional smoke checks for scrypted python plugins.

Each check exercises the third party dependencies the way the plugin does (the calls are modelled
on the plugin sources in koush/scrypted: plugins/<name>/src), using tiny synthetic models/media
so the checks need no model downloads.

usage: smoke_checks.py <plugin|_interpreter> [<pip-target>]

Exit status is 1 if any check FAILs. Checks that cannot run in this environment (missing optional
system libraries, no macOS, ...) are reported as SKIP and do not fail the run.
"""
import base64
import io
import json
import os
import sys
import tempfile
import traceback

plugin = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else None
# the scrypted server does sys.path.append(pip_target) after the interpreter's own packages
if target:
    sys.path.append(target)

results = []


class Skip(Exception):
    pass


def check(fn):
    name = fn.__name__
    try:
        detail = fn()
        results.append((name, "ok", detail or ""))
        print(f"[smoke] {name}: ok {detail or ''}", flush=True)
    except Skip as e:
        results.append((name, "skip", str(e)))
        print(f"[smoke] {name}: SKIP {e}", flush=True)
    except BaseException as e:  # noqa: B902 - native crashes surface as exceptions too
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[smoke] {name}: FAIL {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    return fn


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------------------------
# embedded fixtures (generated once with the onnx and tensorflow packages; opset 13 / int8 full-integer quant)
# ---------------------------------------------------------------------------------------------
# onnx: 1x3x16x16 float32 input -> Conv 1x1 (3->2, weights W below) -> Relu -> output
ONNX_MODEL_B64 = (
    "CAgSBXNtb2tlOrQBCisKBmltYWdlcwoBVxIBYyIEQ29udioVCgxrZXJuZWxfc2hhcGVAAUABoAEH"
    "ChEKAWMSBm91dHB1dCIEUmVsdRIFc21va2UqJwgCCAMIAQgBEAFCAVdKGAAAAD8AAIC/AACAPgAA"
    "gD8AAAA/AAAAv1ogCgZpbWFnZXMSFgoUCAESEAoCCAEKAggDCgIIEAoCCBBiIAoGb3V0cHV0EhYK"
    "FAgBEhAKAggBCgIIAgoCCBAKAggQQgQKABAN"
)
ONNX_CONV_W = [[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]]
# tflite: full integer quantised, int8 1x16x16x3 input -> conv -> int8 output
TFLITE_MODEL_B64 = (
    "HAAAAFRGTDMUACAAHAAYABQAEAAMAAAACAAEABQAAAAcAAAAmAAAAPAAAADQAQAA4AEAAJAEAAAD"
    "AAAAAQAAABAAAAAAAAoAEAAMAAgABAAKAAAADAAAABwAAAA8AAAADwAAAHNlcnZpbmdfZGVmYXVs"
    "dAABAAAABAAAAJD///8DAAAABAAAAAgAAABvdXRwdXRfMAAAAAABAAAABAAAAMr+//8EAAAADAAA"
    "AGtlcmFzX3RlbnNvcgAAAAACAAAANAAAAAQAAADc////BgAAAAQAAAATAAAAQ09OVkVSU0lPTl9N"
    "RVRBREFUQQAIAAwACAAEAAgAAAAFAAAABAAAABMAAABtaW5fcnVudGltZV92ZXJzaW9uAAcAAADc"
    "AAAA0AAAALgAAACcAAAAlAAAAHQAAAAEAAAAYv///wQAAABgAAAADAAAAAgADgAIAAQACAAAABAA"
    "AAAoAAAAAAAGAAgABAAGAAAABAAAAAEAAADrAwAADAAYABQAEAAMAAQADAAAAIDvxx43nkbNAgAA"
    "AAIAAAAEAAAABgAAADIuMjEuMAAAzv///wQAAAAQAAAAMS4xNC4wAAAAAAAAAAAAAMz////u////"
    "BAAAAAYAAABAgSB/QMAGAAgABAAGAAAABAAAAAgAAAAAAAAAAAAAAPz///8EAAQABAAAAA8AAABN"
    "TElSIENvbnZlcnRlZC4AAQAAABQAAAAAAA4AGAAUABAADAAIAAQADgAAABQAAAAcAAAAfAAAAIAA"
    "AACEAAAABAAAAG1haW4AAAAAAQAAABQAAAAAAA4AFAAAABAADAALAAQADgAAABwAAAAAAAABKAAA"
    "ACwAAAAMABQAEwAMAAgABwAMAAAAAAAAAQEAAAABAAAAAAAAAQEAAAADAAAAAwAAAAAAAAACAAAA"
    "AQAAAAEAAAADAAAAAQAAAAAAAAAEAAAAfAEAAPQAAAB4AAAABAAAAKr+//8AAAABFAAAADAAAAAE"
    "AAAAAAAACUQAAACU/v//CAAAABAAAAABAAAAgP////////8BAAAAyT22OxsAAABTdGF0ZWZ1bFBh"
    "cnRpdGlvbmVkQ2FsbF8xOjAABAAAAAEAAAAQAAAAEAAAAAIAAAAa////AAAAARQAAAAwAAAAAwAA"
    "AAAAAAlMAAAABP///wgAAAAQAAAAAQAAAAAAAAAAAAAAAQAAAAQCATwiAAAAZnVuY3Rpb25hbF8x"
    "L2NvbnYyZF8xL2NvbnZvbHV0aW9uMQAABAAAAAIAAAABAAAAAQAAAAMAAACS////AAAAARQAAAAw"
    "AAAAAgAAAAAAAAJMAAAAfP///wgAAAAQAAAAAQAAAAAAAAAAAAAAAQAAAGl/ATghAAAAZnVuY3Rp"
    "b25hbF8xL2NvbnYyZF8xL2NvbnZvbHV0aW9uAAAAAQAAAAIAAAAAABYAHAAYABcAEAAMAAgAAAAA"
    "AAAABwAWAAAAAAAAASAAAABAAAAAAQAAAAAAAAlYAAAADAAMAAAAAAAIAAQADAAAAAgAAAAUAAAA"
    "AQAAAID/////////AAAAAAEAAABqfIA7HgAAAHNlcnZpbmdfZGVmYXVsdF9rZXJhc190ZW5zb3I6"
    "MAAABAAAAAEAAAAQAAAAEAAAAAMAAAABAAAAEAAAAAwAEAAPAAAACAAEAAwAAAADAAAAAwAAAAAA"
    "AAM="
)


def tmpfile(name, data):
    d = tempfile.mkdtemp(prefix="smoke-")
    p = os.path.join(d, name)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(p, mode) as f:
        f.write(data)
    return p


# ---------------------------------------------------------------------------------------------
# interpreter
# ---------------------------------------------------------------------------------------------
if plugin == "_interpreter":

    @check
    def stdlib_native_modules():
        import bz2, ctypes, hashlib, lzma, sqlite3, ssl, zlib  # noqa: E401

        need(zlib.decompress(zlib.compress(b"x" * 1000)) == b"x" * 1000, "zlib")
        need(bz2.decompress(bz2.compress(b"y" * 1000)) == b"y" * 1000, "bz2")
        need(lzma.decompress(lzma.compress(b"z" * 1000)) == b"z" * 1000, "lzma")
        need(hashlib.sha256(b"abc").hexdigest().startswith("ba7816bf"), "sha256")
        db = sqlite3.connect(":memory:")
        db.execute("create table t(a)")
        db.execute("insert into t values (?)", (1,))
        need(db.execute("select a from t").fetchone() == (1,), "sqlite3")
        need(ctypes.c_int(5).value == 5 and ctypes.sizeof(ctypes.c_void_p) in (4, 8), "ctypes")
        return f"{ssl.OPENSSL_VERSION}; sqlite {sqlite3.sqlite_version}"

    @check
    def https_request():
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        with urllib.request.urlopen("https://pypi.org/simple/pip/", context=ctx, timeout=30) as r:
            need(r.status == 200, f"status {r.status}")
        return "pypi.org ok"

    @check
    def asyncio_threads_subprocess():
        import asyncio
        import concurrent.futures
        import subprocess

        async def main():
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(2, "smoke") as ex:
                return await loop.run_in_executor(ex, lambda: 6 * 7)

        need(asyncio.run(main()) == 42, "executor")
        out = subprocess.check_output([sys.executable, "-c", "print('child ok')"], text=True)
        need(out.strip() == "child ok", "subprocess")

    @check
    def pip_works():
        import subprocess

        out = subprocess.check_output([sys.executable, "-m", "pip", "--version"], text=True)
        return out.split()[1]


# ---------------------------------------------------------------------------------------------
# opencv  (plugins/opencv/src/opencv/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "opencv":

    @check
    def motion_detection_pipeline():
        # mirrors OpenCVPlugin.detect: blur -> absdiff -> threshold -> dilate -> findContours
        import cv2
        import imutils
        import numpy as np

        a = np.zeros((120, 160), dtype=np.uint8)
        b = a.copy()
        b[40:80, 60:100] = 255
        ba = cv2.GaussianBlur(a, (5, 5), 0)
        bb = cv2.GaussianBlur(b, (5, 5), 0)
        delta = cv2.absdiff(ba, bb)
        thresh = np.zeros_like(delta)
        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY, dst=thresh)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours = imutils.grab_contours(
            cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )
        need(len(contours) == 1, f"expected 1 contour, got {len(contours)}")
        x, y, w, h = cv2.boundingRect(contours[0])
        need(abs(x - 60) <= 6 and abs(y - 40) <= 6 and abs(w - 40) <= 12 and abs(h - 40) <= 12,
             f"bad box {(x, y, w, h)}")
        return f"opencv {cv2.__version__} box={(x, y, w, h)}"

    @check
    def pil_and_buffers():
        # mirrors the Image.frombuffer / resize / np.ndarray(buffer=) handling in the plugin
        import numpy as np
        from PIL import Image

        w, h = 48, 24
        rgb = bytes(range(256)) * (w * h * 3 // 256 + 1)
        rgb = rgb[: w * h * 3]
        img = Image.frombuffer("RGB", (w, h), rgb)
        small = img.resize((24, 12), Image.BILINEAR)
        gray = Image.frombuffer("L", (w, h), rgb[: w * h])
        mat = np.ndarray((12, 24, 3), buffer=small.tobytes(), dtype=np.uint8)
        need(mat.shape == (12, 24, 3) and mat.any(), "ndarray from PIL buffer")
        need(gray.size == (w, h), "L image")

    @check
    def video_roundtrip():
        # the plugin reads frames with cv2.VideoCapture
        import cv2
        import numpy as np

        path = os.path.join(tempfile.mkdtemp(prefix="smoke-"), "t.avi")
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        need(wr.isOpened(), "VideoWriter did not open (MJPG/avi)")
        for i in range(10):
            wr.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        wr.release()
        cap = cv2.VideoCapture(path)
        need(cap.isOpened(), "VideoCapture did not open")
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            need(frame.shape == (48, 64, 3), f"frame shape {frame.shape}")
            n += 1
        cap.release()
        need(n == 10, f"read {n} frames, expected 10")


# ---------------------------------------------------------------------------------------------
# onnx  (plugins/onnx/src/ort/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "onnx":

    @check
    def inference_session():
        import numpy as np
        import onnxruntime

        model = tmpfile("model.onnx", base64.b64decode(ONNX_MODEL_B64))
        sess_options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        sess = onnxruntime.InferenceSession(model, sess_options=sess_options, providers=providers)
        inp = sess.get_inputs()[0]
        need(list(inp.shape) == [1, 3, 16, 16], f"input shape {inp.shape}")
        x = np.random.RandomState(0).rand(1, 3, 16, 16).astype(np.float32)
        (out,) = sess.run(None, {inp.name: x})
        w = np.array(ONNX_CONV_W, dtype=np.float32)
        want = np.maximum(np.einsum("oc,nchw->nohw", w, x), 0)
        need(np.allclose(out, want, atol=1e-5), "output mismatch vs numpy reference")
        return f"onnxruntime {onnxruntime.__version__} providers={onnxruntime.get_available_providers()}"

    @check
    def clip_processor_import():
        # plugins/onnx/src/predict/clip.py -> shared with openvino
        from transformers import CLIPProcessor  # noqa: F401


# ---------------------------------------------------------------------------------------------
# openvino  (plugins/openvino/src/ov/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "openvino":

    @check
    def core_and_devices():
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        need("CPU" in devices, f"no CPU device in {devices}")
        for device in devices:
            core.get_property(device, "FULL_DEVICE_NAME")
            for key in core.get_property(device, "SUPPORTED_PROPERTIES"):
                pass
        return f"openvino {ov.__version__} devices={devices}"

    @check
    def compile_and_async_infer():
        # mirrors: core.compile_model(xml, device); AsyncInferQueue + callback + start_async
        import asyncio

        import numpy as np
        import openvino as ov
        import openvino.opset13 as ops

        param = ops.parameter([1, 3, 16, 16], np.float32, name="images")
        w = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]], dtype=np.float32).reshape(2, 3, 1, 1)
        conv = ops.convolution(param, ops.constant(w), [1, 1], [0, 0], [0, 0], [1, 1])
        model = ov.Model([ops.relu(conv)], [param], "smoke")
        xml = os.path.join(tempfile.mkdtemp(prefix="smoke-"), "model.xml")
        ov.save_model(model, xml)

        core = ov.Core()
        compiled = core.compile_model(xml, "CPU")
        infer_queue = ov.AsyncInferQueue(compiled)

        async def run():
            loop = asyncio.get_event_loop()
            fut = loop.create_future()

            def callback(infer_request, userdata):
                try:
                    out = infer_request.get_output_tensor(0).data
                    loop.call_soon_threadsafe(fut.set_result, np.array(out))
                except Exception as e:
                    loop.call_soon_threadsafe(fut.set_exception, e)

            infer_queue.set_callback(callback)
            x = np.random.RandomState(0).rand(1, 3, 16, 16).astype(np.float32)
            infer_queue.start_async(ov.Tensor(array=x), fut)
            out = await asyncio.wait_for(fut, 30)
            return x, out

        x, out = asyncio.run(run())
        want = np.maximum(np.einsum("oc,nchw->nohw", w.reshape(2, 3), x), 0)
        # the CPU plugin runs in bfloat16 by default on CPUs that support it, hence the tolerance
        need(np.allclose(out, want, atol=2e-2, rtol=2e-2), f"output mismatch vs numpy reference, max err {np.abs(out - want).max()}")

    @check
    def clip_processor_import():
        # plugins/openvino/src/predict/clip.py
        from transformers import CLIPProcessor  # noqa: F401

    @check
    def cv2_and_pil():
        import cv2
        import numpy as np
        from PIL import Image

        im = Image.fromarray(np.zeros((8, 8, 3), np.uint8)).resize((16, 16), Image.BILINEAR)
        need(cv2.resize(np.array(im), (4, 4)).shape == (4, 4, 3), "cv2.resize")


# ---------------------------------------------------------------------------------------------
# ncnn  (plugins/ncnn/src/nc/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "ncnn":

    @check
    def net_inference():
        import struct

        import ncnn
        import numpy as np

        # 1x1 convolution 3 -> 2 channels followed by ReLU; weights are loaded from the .bin like a
        # real model. Weights use ncnn's raw float32 encoding: 4 zero bytes (type flag) then floats.
        w = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]], dtype=np.float32)
        param = "\n".join(
            [
                "7767517",
                "3 3",
                "Input            data   0 1 data",
                "Convolution      conv   1 1 data conv0 0=2 1=1 5=0 6=6",
                "ReLU             relu   1 1 conv0 out0",
            ]
        )
        d = tempfile.mkdtemp(prefix="smoke-")
        param_file = os.path.join(d, "model.ncnn.param")
        bin_file = os.path.join(d, "model.ncnn.bin")
        open(param_file, "w").write(param + "\n")
        open(bin_file, "wb").write(struct.pack("<I", 0) + w.tobytes())

        net = ncnn.Net()
        net.opt.use_vulkan_compute = True  # as in the plugin; falls back to CPU without vulkan
        need(net.load_param(param_file) == 0, "load_param")
        need(net.load_model(bin_file) == 0, "load_model")

        x = np.random.RandomState(0).rand(3, 16, 16).astype(np.float32)
        ex = net.create_extractor()
        ex.input("data", ncnn.Mat(np.ascontiguousarray(x)))
        out = ncnn.Mat()
        ex.extract("out0", out)
        got = np.array(out)
        want = np.maximum(np.einsum("oc,chw->ohw", w, x), 0)
        need(got.shape == want.shape, f"shape {got.shape} vs {want.shape}")
        need(np.allclose(got, want, atol=1e-4), "output mismatch vs numpy reference")
        return f"ncnn {ncnn.__version__}"

    @check
    def huggingface_hub_import():
        import huggingface_hub  # noqa: F401

    @check
    def cv2_and_pil():
        import cv2
        import numpy as np
        from PIL import Image

        im = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
        need(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY).shape == (8, 8), "cvtColor")


# ---------------------------------------------------------------------------------------------
# coreml  (plugins/coreml/src/coreml/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "coreml":

    @check
    def build_convert_and_predict():
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb

        w = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]], dtype=np.float32).reshape(2, 3, 1, 1)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 16, 16))])
        def prog(x):
            y = mb.conv(x=x, weight=w)
            return mb.relu(x=y, name="out0")

        mlmodel = ct.convert(prog, convert_to="mlprogram")
        need(mlmodel is not None, "convert returned nothing")
        if sys.platform != "darwin":
            # CoreML inference is only available on macOS
            raise Skip(f"coremltools {ct.__version__}: model built/converted, predict needs macOS")
        path = os.path.join(tempfile.mkdtemp(prefix="smoke-"), "model.mlpackage")
        mlmodel.save(path)
        model = ct.models.MLModel(path)  # as the plugin does with a model file
        inp = model.get_spec().description.input[0].name
        x = np.random.RandomState(0).rand(1, 3, 16, 16).astype(np.float32)
        out = list(model.predict({inp: x}).values())[0]
        want = np.maximum(np.einsum("oc,nchw->nohw", w.reshape(2, 3), x), 0)
        need(np.allclose(out, want, atol=1e-2), "output mismatch vs numpy reference")
        return f"coremltools {ct.__version__}"

    @check
    def clip_processor_import():
        from transformers import CLIPProcessor  # noqa: F401

    @check
    def cv2_and_pil():
        import cv2
        import numpy as np
        from PIL import Image

        im = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
        need(cv2.resize(np.array(im), (4, 4)).shape == (4, 4, 3), "cv2.resize")


# ---------------------------------------------------------------------------------------------
# tensorflow-lite  (plugins/tensorflow-lite/src/tflite/__init__.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "tensorflow-lite":

    @check
    def interpreter_invoke():
        import numpy as np
        import tflite_runtime.interpreter as tflite

        model = tmpfile("model.tflite", base64.b64decode(TFLITE_MODEL_B64))
        interpreter = tflite.Interpreter(model_path=model)
        try:
            interpreter.allocate_tensors()
        except RuntimeError as e:
            # the fixture is written by a recent TensorFlow; the pinned tflite-runtime is from 2021
            if "Didn't find op for builtin opcode" in str(e):
                raise Skip(f"fixture uses ops this old tflite-runtime lacks: {str(e).splitlines()[0]}")
            raise
        details = interpreter.get_input_details()[0]
        need(list(details["shape"]) == [1, 16, 16, 3], f"input shape {details['shape']}")
        scale, zero_point = details["quantization"]
        # same int8 quantisation the plugin applies to the incoming uint8 image
        im = np.random.RandomState(0).randint(0, 255, (1, 16, 16, 3)).astype(np.uint8)
        q = im.astype(np.float32) / (255.0 * scale)
        q = (q + zero_point).astype(np.int8)
        interpreter.set_tensor(details["index"], q)
        interpreter.invoke()
        outs = [
            (interpreter.get_tensor(o["index"]), o) for o in interpreter.get_output_details()
        ]
        need(len(outs) >= 1 and outs[0][0].size > 0, "no output tensor")
        out, od = outs[0]
        need(out.dtype == np.int8 and list(out.shape) == [1, 16, 16, 2], f"output {out.dtype} {out.shape}")
        # dequantise input and output and compare with the float reference
        w = np.array(ONNX_CONV_W, dtype=np.float32)
        xr = (q.astype(np.float32) - zero_point) * scale
        want = np.maximum(np.einsum("oc,nhwc->nhwo", w, xr), 0)
        oscale, ozero = od["quantization"]
        got = (out.astype(np.float32) - ozero) * oscale
        need(np.abs(got - want).max() < 0.05, f"dequantised output off by {np.abs(got - want).max()}")
        return f"max err {np.abs(got - want).max():.4f}"

    @check
    def pycoral_without_tpu():
        # plugin: from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
        except ImportError as e:
            raise Skip(f"pycoral not importable here: {e}")
        tpus = list_edge_tpus()
        need(isinstance(tpus, list), f"list_edge_tpus returned {type(tpus)}")
        return f"edge tpus: {len(tpus)}"

    @check
    def pil_numpy():
        import numpy as np
        from PIL import Image

        im = Image.fromarray(np.zeros((8, 8, 3), np.uint8)).resize((4, 4), Image.BILINEAR)
        need(np.array(im).shape == (4, 4, 3), "PIL resize")


# ---------------------------------------------------------------------------------------------
# python-codecs  (plugins/python-codecs/src/{libav,pilimage,vipsimage,gstreamer}.py)
# ---------------------------------------------------------------------------------------------
elif plugin == "python-codecs":

    @check
    def libav_decode():
        # mirrors generateVideoFramesLibav: av.open -> streams.video[0] -> decode -> to_ndarray/to_image
        import av
        import numpy as np

        path = os.path.join(tempfile.mkdtemp(prefix="smoke-"), "t.mp4")
        out = av.open(path, "w")
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()

        av.logging.set_level(av.logging.PANIC)
        container = av.open(path)
        container.options["analyzeduration"] = "0"
        container.options["probesize"] = "500000"
        vs = container.streams.video[0]
        n = 0
        for frame in container.decode(vs):
            gray = frame.to_ndarray(format="gray")
            rgb = frame.to_ndarray(format="rgb24")
            pil = frame.to_image()
            need(gray.shape == (48, 64) and rgb.shape == (48, 64, 3) and pil.size == (64, 48), "frame shapes")
            n += 1
        container.close()
        need(n == 10, f"decoded {n} frames, expected 10")
        return f"av {av.__version__}"

    @check
    def pil_image_ops():
        # mirrors pilimage.py: frombuffer, crop, resize, convert, tobytes, JPEG save/open
        from PIL import Image

        w, h = 64, 48
        data = bytes((i * 7) % 256 for i in range(w * h * 4))
        for mode, bands in (("RGBA", 4), ("RGB", 3), ("L", 1)):
            img = Image.frombuffer(mode, (w, h), data[: w * h * bands])
            need(img.width == w and img.height == h, f"{mode} frombuffer")
        img = Image.frombuffer("RGBA", (w, h), data)
        img = img.crop((8, 8, 8 + 32, 8 + 24)).resize((16, 12), resample=Image.BILINEAR)
        rgb = img.convert("RGB")
        need(len(rgb.tobytes()) == 16 * 12 * 3, "tobytes")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG")
        need(Image.open(io.BytesIO(buf.getvalue())).size == (16, 12), "jpeg roundtrip")

    @check
    def pyvips_image_ops():
        # mirrors vipsimage.py; pyvips needs the libvips system library, which scrypted's installers provide
        try:
            import pyvips
            from pyvips import Image
        except (ImportError, OSError) as e:
            raise Skip(f"pyvips unavailable (needs system libvips): {str(e).splitlines()[0]}")
        w, h = 64, 48
        data = bytes((i * 5) % 256 for i in range(w * h * 3))
        vips = Image.new_from_memory(data, w, h, 3, pyvips.BandFormat.UCHAR)
        vips = vips.crop(8, 8, 32, 24)
        vips = vips.resize(0.5, vscale=0.5, kernel="linear")
        need(vips.width == 16 and vips.height == 12, f"size {vips.width}x{vips.height}")
        mem = bytes(memoryview(vips.write_to_memory()))
        need(len(mem) == 16 * 12 * 3, "write_to_memory")
        jpg = vips.write_to_buffer(".jpg[Q=80]")
        need(Image.new_from_buffer(jpg, "").width == 16, "jpeg roundtrip")

    @check
    def gstreamer_bindings():
        # gstreamer.py / gst_generator.py use PyGObject; optional and needs system GStreamer
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
        except Exception as e:
            raise Skip(f"PyGObject/GStreamer unavailable: {str(e).splitlines()[0]}")
        return f"Gst {Gst.version_string()}"


# ---------------------------------------------------------------------------------------------
# wyze  (plugins/wyze/src/main.py + docker-wyze-bridge wyzecam: requests, pydantic, xxtea)
# ---------------------------------------------------------------------------------------------
elif plugin == "wyze":

    @check
    def xxtea_roundtrip():
        import xxtea

        key = b"0123456789abcdef"
        msg = b"wyze tutk payload " * 5
        enc = xxtea.encrypt(msg, key)
        need(enc != msg and xxtea.decrypt(enc, key) == msg, "xxtea roundtrip")

    @check
    def pydantic_models():
        from typing import Optional

        import pydantic
        from pydantic import BaseModel

        class Device(BaseModel):
            mac: str
            nickname: str = ""
            product_model: str
            enr: Optional[str] = None

        d = Device.model_validate({"mac": "AA", "product_model": "WYZEC1"}) if hasattr(
            Device, "model_validate"
        ) else Device.parse_obj({"mac": "AA", "product_model": "WYZEC1"})
        need(d.mac == "AA" and d.nickname == "", "defaults")
        try:
            Device(mac=1.5, product_model=None)  # invalid
        except pydantic.ValidationError:
            pass
        else:
            raise AssertionError("expected ValidationError")
        return f"pydantic {pydantic.VERSION}"

    @check
    def requests_http():
        # wyzecam.api uses requests (post_device, HTTPError, RequestException); use a local server
        import http.server
        import threading

        import requests
        from requests import HTTPError, RequestException  # noqa: F401

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                if self.path == "/fail":
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"echo": json.loads(body)}).encode())

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            r = requests.post(base + "/ok", json={"a": 1}, timeout=10)
            need(r.json() == {"echo": {"a": 1}}, "json echo")
            try:
                requests.post(base + "/fail", json={}, timeout=10).raise_for_status()
            except HTTPError:
                pass
            else:
                raise AssertionError("expected HTTPError")
        finally:
            srv.shutdown()

    @check
    def ctypes_struct():
        # main.py: from ctypes import c_int (tutk bindings are ctypes based)
        from ctypes import Structure, c_int, c_ubyte, sizeof

        class Hdr(Structure):
            _fields_ = [("a", c_int), ("b", c_ubyte)]

        need(sizeof(Hdr) >= 5, "struct size")


else:
    print(f"[smoke] no checks defined for {plugin}")


# ---------------------------------------------------------------------------------------------
summary = {"plugin": plugin, "python": sys.version.split()[0], "checks": [
    {"name": n, "status": s, "detail": d} for n, s, d in results
]}
out = os.environ.get("SMOKE_JSON")
if out:
    with open(out, "w") as f:
        json.dump(summary, f)
failed = [n for n, s, _ in results if s == "FAIL"]
skipped = [n for n, s, _ in results if s == "skip"]
print(f"[smoke] {plugin}: {len(results) - len(failed) - len(skipped)} ok, {len(skipped)} skipped, {len(failed)} failed")
sys.exit(1 if failed else 0)
