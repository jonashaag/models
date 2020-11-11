import numpy as np
import base64
import json

from werkzeug.wrappers import Request, Response

import params as yamnet_params
import yamnet as yamnet_model


DEFAULT_TOP_N = 5


def decode_audio(audio_bytes):
    return np.frombuffer(base64.b64decode(audio_bytes), dtype="float32")


def make_app(make_predict_func):
    predict_func = None

    def app(environ, start_response):
        nonlocal predict_func
        if predict_func is None:
            predict_func = make_predict_func()

        request = Request(environ)
        inputs = json.loads(request.get_data())
        top_n = int(request.args.get('top_n', 0)) or None

        outputs = []
        for inp in inputs:
            try:
                pred = predict_func(decode_audio(inp), top_n)
            except Exception as e:
                print(f"Error predicting classes for input {len(outputs)}: {e}")
                pred = None
            outputs.append(pred)

        return Response(json.dumps(outputs))(environ, start_response)

    return app


def load_model():
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
    return yamnet, yamnet_classes


def predict_classes(audio, top_n=None, *, model, labels):
    scores, embeddings, spectrogram = model(audio)
    prediction = np.mean(scores, axis=0)
    idxs_by_score = np.argsort(prediction)[::-1]
    if not top_n:
        top_n = DEFAULT_TOP_N
    if top_n:
        idxs_by_score = idxs_by_score[:top_n]
    return [(labels[i], float(prediction[i])) for i in idxs_by_score]


if __name__ == "__main__":
    import argparse
    import functools
    import bjoern
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str, help="host to serve")
    parser.add_argument("--port", required=True, type=int, help="port to serve")
    parser.add_argument("--processes", default=1, type=int, help="number of processes to spawn")
    args = parser.parse_args()

    def make_predict_func():
        model, labels = load_model()
        return functools.partial(predict_classes, model=model, labels=labels)

    app = make_app(make_predict_func)
    procs = [multiprocessing.Process(target=functools.partial(bjoern.run, app, args.host, args.port, reuse_port=True))
             for i in range(args.processes)]
    for p in procs:
        p.start()
    print("Ready")
    for p in procs:
        p.join()
