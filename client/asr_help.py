import json


def raw_json_to_non_json(d: dict):
    if len(d['segments']) < 1:
        return []
    result_list = []
    s = d['segments'][0]["start"] * 100
    e = d['segments'][-1]["end"] * 100
    result_list.append({
        'audio_offset': s,
        "boundary_type": "sentence",
        'duration': e - s,
        'test_offset': len(d['text']),
        'text': d['text'],
        'word_length': len(d['text'])
    })
    text_offset = len(d['text'])
    for segment in d['segments']:
        for word in segment['words']:
            result_list.append({
                'audio_offset': word['start'] * 100,
                "boundary_type": "word",
                'duration': (word['end'] - word['start']) * 100,
                'test_offset': text_offset,
                'text': word['word'],
                'word_length': len(word['word'])
            })
            text_offset = text_offset + len(word['word'])
    return json.dumps(result_list)

def to_vtt_data(raw_file ,result :any):
    # TODO: 这个可能需要重新实现
    cfile = raw_file + ".vtt"
    result.to_srt_vtt(cfile)
    with open(cfile, "r", encoding="utf-8") as f:
        d = f.read()
    # os.remove(inpf)
    # os.remove(cfile)
    return d
