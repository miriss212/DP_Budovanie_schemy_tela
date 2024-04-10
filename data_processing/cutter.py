import json

def make_chunk(samplesCount):

    data_source = "my_data_act.data"
    f = open(data_source)
    data = json.load(f)
    f.close()

    data_cut = data[:samplesCount]

    f = open("data_chunk.data", "w")
    f.write(str(json.dumps(data_cut, indent=4)))
    f.close()


make_chunk(1000)
