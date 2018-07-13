#!/usr/bin/python
# -*- coding: utf8 -*-

import sys

def read_file(input_file):
	simple_sen = []
	complex_sen = []
	all_sentences = []
	with open(input_file) as i:
		for line in i.readlines():
			line = line.strip()
			if line != "*******":
				all_sentences.append(line)
	for i, sentence in enumerate(all_sentences):
		if i%2 == 0:
			complex_sen.append(sentence)
		else:
			simple_sen.append(sentence)
	return complex_sen, simple_sen

def write_to_file(sentences, output):
	with open(output, 'w') as out:
		for line in sentences:
			out.write("{}\n".format(line))

def main(argv):
	if len(argv) != 4:
		print("usage: {} one_stop_english output_complex output_simple".format(argv[0]))
		sys.exit(-1)
	complex_sen, simple_sen = read_file(argv[1])
	write_to_file(complex_sen, argv[2])
	write_to_file(simple_sen, argv[3])

if __name__ == '__main__':
	main(sys.argv)