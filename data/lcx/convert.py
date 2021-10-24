import json

def convert(path:str):
    with open(path, 'r') as f:
        data = json.load(f)
    output_path = path.replace('convert', 'triplets').replace('json', 'txt')
    with open(output_path, 'w') as f:
        for ins in data:
            temp_ins = []
            for a, o in zip(ins['aspects'], ins['opinions']):
                temp_ins.append(([x for x in range(a['from'], a['to'])], [x for x in range(o['from'], o['to'])], a['polarity']))
            f.write(ins['raw_words'] + '####' + str(temp_ins) + '\n')
    return

if __name__ == '__main__':
    convert('14lap/train_convert.json')
    convert('14lap/test_convert.json')
    convert('14res/train_convert.json')
    convert('14res/test_convert.json')
    convert('15res/train_convert.json')
    convert('15res/test_convert.json')
    convert('16res/train_convert.json')
    convert('16res/test_convert.json')