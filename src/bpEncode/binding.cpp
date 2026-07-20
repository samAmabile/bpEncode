#include "bpEncode/bpe.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(_bpEncode, m) {
    m.doc() = "Byte-Pair Encoding Tokenizer";

    pybind11::class_<BPE>(m, "BPE")
        .def(pybind11::init<>())
        .def("reset", &BPE::reset, "reset all data in model")
        .def("setVocabSize", &BPE::setVocabSize, "Sets the upper limit for unique tokens")
        .def("initVocab", &BPE::initVocab, "Initializes utf-8 tokens (0 - 255)")
        .def("fitText", &BPE::fitText, "fits vocab based on byte-pairs of string")
        .def("preTokenize", &BPE::preTokenize, "encodes text respecting word boundaries")
        .def("fitWords", &BPE::fitWords, "fits vocab based on sub-word byte-pair tokens")
        .def("tokenize", &BPE::tokenize, "creates a bytepair encoded list of tokens from a string based on vocab from fit")
        .def("decode", &BPE::decode, "converts bytepair tokens back into the source string based on current encoding")
        .def("getVocabSize", &BPE::getVocabSize, "returns the number of unique tokens in the vocab")
        .def("save", &BPE::save, "save merge rules to binary file to encode/decode future files that use the same scheme")
        .def("load", &BPE::load, "load merge rules to encode/decode files with a previously saved scheme");
}

