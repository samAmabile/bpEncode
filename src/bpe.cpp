#include "bpe.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>


void BPE::setVocabSize(const size_t limit){ 
    LIMIT = limit; 
}

void BPE::setMinFreq(const size_t freq){
    minFreq = freq; 
}

void BPE::reset(){
    mergeRules.clear(); 
    mergeHistory.clear();
    vocab.clear(); 
    LIMIT = 0; 
    buffer.clear(); 
    positions.clear(); 
    pairQueue = std::priority_queue<std::pair<size_t, uint64_t>>();
}

std::vector<uint32_t> BPE::preTokenize(const std::string& text){

    initVocab();

    std::vector<uint32_t> encodedText; 
    std::string word; 

    encodedText.push_back(WORD_START);

    for (char c: text){
        if (isspace(c)){
            encodedText.push_back(WORD_END); 
            encodedText.push_back(SPACE); 
            encodedText.push_back(WORD_START); 
        } else{
            encodedText.push_back(static_cast<uint32_t>(c));
        }

        encodedText.push_back(WORD_END); 

    }

    return encodedText;
    
}

void BPE::initVocab(){
    for (size_t i{0}; i < 256; i++){
        vocab[i] = {static_cast<uint8_t>(i)}; 
    }
}

std::vector<uint32_t> BPE::encodeStr(const std::string& text){
    
    std::vector<uint32_t> buf; 
    for (const unsigned char c: text){ 
        buf.push_back(static_cast<uint32_t>(c)); 
    }

    return buf; 
}

uint32_t BPE::getVocabSize() const {
    return static_cast<uint32_t>(vocab.size()); 
}

uint64_t BPE::countPairs(std::vector<uint32_t>& buffer){
    size_t i = 0; 
    size_t j = 1; 
    //freqDist pairCounts;
    size_t max_count = 0; 
    uint64_t max_pair = 0; 
    while (j < buffer.size()){
        uint64_t pair = (static_cast<uint64_t>(buffer[i]) << 32 | buffer[j]); 
        pairCounts[pair]++; 
        pairQueue.push({pairCounts[pair], pair});
        if (pairCounts[pair] > max_count){
            max_count = pairCounts[pair];
            max_pair = pair; 
        }
        i++; j++; 
    }
    if (max_count <= minFreq) LIMIT = getVocabSize()-1; 
    return max_pair;
}

uint64_t BPE::getBestPair() {
    while(!pairQueue.empty()){
        auto top = pairQueue.top();
        uint64_t pair = top.second;

        if (pairCounts[pair] == top.first){
            return pair; 
        }

        pairQueue.pop();

    }

    return 0; 
}
void BPE::countSubwordPairs(std::vector<uint32_t>& buffer){
    //freqDist pairCounts; //this is a member variable now
    //std::priority_queue<std::pair<size_t, uint64_t>> pairQueue; //this lives in private member variables of class and persists outside this function

    pairCounts.clear();
    pairQueue = std::priority_queue<std::pair<size_t, uint64_t>>();
    positions.clear();

    size_t max_count = 0; 
    uint64_t max_pair = 0; 
    
    for (size_t i{0}; i < buffer.size()-1; ++i){
        uint32_t k = buffer[i]; 
        uint32_t j = buffer[i + 1];

        if (k == WORD_START || k == WORD_END || k == SPACE ||
            j == WORD_START || j == WORD_END || j == SPACE) {
            continue;
        }
        
        positions[k].push_back(i); 

        uint64_t pair = (static_cast<uint64_t>(k) << 32) | j; 
        pairCounts[pair]++; 
        pairQueue.push({pairCounts[pair], pair}); 

    }

}

void BPE::updateVocab(const uint64_t key){
    uint32_t id1 = static_cast<uint32_t>(key >> 32); 
    uint32_t id2 = static_cast<uint32_t>(key & 0xFFFFFFFF);//11111...X32 

    std::vector<uint8_t> bytes1 = vocab[id1]; 
    std::vector<uint8_t> bytes2 = vocab[id2];

    std::vector<uint8_t> merged = bytes1; 
    merged.insert(merged.end(), bytes2.begin(), bytes2.end()); 
    uint32_t newID = getVocabSize(); 
    vocab[newID] = merged; 
    updateBuffer(id1, id2, newID);


    mergeHistory.push_back({id1, id2});
    mergeRules[{id1, id2}] = newID;
    //mergeMap[key] = newID;

}

size_t BPE::findNextValidIndex(size_t index){
    for (size_t i = index + 1; i < buffer.size(); ++i){
        if (buffer[i] != INVALID_TOKEN) {
            return i; 
        }
    }
    return static_cast<size_t>(-1); //not found
}

size_t BPE::findPrevValidIndex(size_t index){
    for (int i = static_cast<int>(index) - 1; i >= 0; --i){
        if (buffer[i] != INVALID_TOKEN) return i; 
    }
    return static_cast<size_t>(-1); 
}


void BPE::updateBuffer(const uint32_t id1, const uint32_t id2, const uint32_t newID){

    auto& indices = positions[id1];

    std::vector<size_t> successfulMerges; 
    std::vector<size_t> remainingIndices;

    for (size_t i : indices) {
        size_t next_i = findNextValidIndex(i); 

        if (next_i != -1 && buffer[next_i] == id2) {

            uint64_t consumedPair = (static_cast<uint64_t>(id1) << 32) | id2;
            pairCounts[consumedPair]--;


            size_t prev_i = findPrevValidIndex(i); 
            size_t next_next_i = findNextValidIndex(next_i); 

            if (prev_i != -1){
                uint64_t old_pair = (static_cast<uint64_t>(buffer[prev_i]) << 32) | id1; 
                pairCounts[old_pair]--; 
                pairQueue.push({pairCounts[old_pair], old_pair}); 
            }
            if (next_next_i != -1){
                uint64_t old_pair = (static_cast<uint64_t>(id2) << 32) | buffer[next_next_i];
                pairCounts[old_pair]--; 
                pairQueue.push({pairCounts[old_pair], old_pair}); 
            }

            buffer[i] = newID; 
            buffer[next_i] = INVALID_TOKEN; 
        
            successfulMerges.push_back(i); 

            if (prev_i != -1){
                uint64_t new_pair = (static_cast<uint64_t>(buffer[prev_i]) << 32) | newID; 
                pairCounts[new_pair]++; 
                pairQueue.push({pairCounts[new_pair], new_pair}); 
            }
            if (next_next_i != -1) { 
                uint64_t new_pair = (static_cast<uint64_t>(newID) << 32) | buffer[next_next_i];
                pairCounts[new_pair]++; 
                pairQueue.push({pairCounts[new_pair], new_pair}); 
            }
        } else {
            remainingIndices.push_back(i);
        }
    }

    positions[newID].insert(positions[newID].end(), successfulMerges.begin(), successfulMerges.end());

    positions[id1] = std::move(remainingIndices);

    /*for (size_t idx : successfulMerges) {
        positions[newID].push_back(idx); 
    }*/
          
}

byte_map BPE::fitWords(const std::string& text){
    buffer = preTokenize(text); 
    uint32_t vocabSize = getVocabSize(); 
    countSubwordPairs(buffer);
    while (vocabSize <= LIMIT){
        uint64_t nextPair = getBestPair(); 
        if (vocabSize >= LIMIT) break;
        if (nextPair == 0) break;
        updateVocab(nextPair); 
        vocabSize = getVocabSize();
    }

    return vocab;
}

byte_map BPE::fitText(const std::string& text){
    buffer = encodeStr(text); 
    uint32_t vocabSize = getVocabSize(); 
    countSubwordPairs(buffer);
    while (vocabSize <= LIMIT) {
        uint64_t nextPair = getBestPair(); 
        if (vocabSize >= LIMIT) break; 
        if (nextPair == 0) break;
        updateVocab(nextPair);
        vocabSize = getVocabSize(); 

    }

    return vocab; 
}



std::vector<uint32_t> BPE::tokenize(const std::string& text){
    std::vector<uint32_t> newBuffer = encodeStr(text); 
    std::vector<uint32_t> bufferCopy; 
    bufferCopy.reserve(newBuffer.size());
    
    for (const auto& pair : mergeHistory){ //make mergeRules mergeHistory
        uint32_t id1 = pair.first;
        uint32_t id2 = pair.second; 
        uint32_t newID = mergeRules.at(pair); //make mergeRules mergeRules

        bufferCopy.clear(); 
        for (size_t i = 0; i < newBuffer.size(); ++i){
            if (i < newBuffer.size()-1 && 
                    newBuffer[i] == id1 &&
                    newBuffer[i+1] == id2){
                
                bufferCopy.push_back(newID); 
                i++;

            }else {

                bufferCopy.push_back(newBuffer[i]); 

            }
        }
        std::swap(newBuffer, bufferCopy); 
    }
    return newBuffer; 
}


std::vector<uint32_t> BPE::tokenizeWords(const std::string& text){
    std::vector<uint32_t> newBuffer = preTokenize(text); 
    
    for (const auto& pair : mergeHistory){ //make mergeRules mergeHistory
        uint32_t id1 = pair.first;
        uint32_t id2 = pair.second; 
        uint32_t newID = mergeRules.at(pair); //make mergeRules mergeRules

        std::vector<uint32_t> bufferCopy; 
        for (size_t i = 0; i < newBuffer.size(); ++i){
            if (newBuffer[i] == WORD_START || newBuffer[i] == WORD_END || newBuffer[i] == SPACE){
                bufferCopy.push_back(newBuffer[i]);
                continue;
            }
            if (i < newBuffer.size()-1 && 
                    newBuffer[i] == id1 &&
                    newBuffer[i+1] == id2){
                
                bufferCopy.push_back(newID); 
                i++;

            }else {

                bufferCopy.push_back(newBuffer[i]); 

            }
        }
        newBuffer = bufferCopy; 
    }
    return newBuffer; 
}



std::string BPE::decode(const std::vector<uint32_t>& byteBuffer){ 

    std::vector<uint8_t> bytes; 

    for (uint32_t byte : byteBuffer){
        bytes.insert(bytes.end(), vocab[byte].begin(), vocab[byte].end());
    }

    std::string originalText(bytes.begin(), bytes.end());

    return originalText; 

}

void BPE::save(const std::string& filename){
    std::ofstream outfile(filename, std::ios::binary);

    //uint32_t magicNumber = 0x42504531; //hex for BPE1

    outfile.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber)); 

    size_t numMerges = mergeHistory.size(); 
    outfile.write(reinterpret_cast<const char*>(&numMerges), sizeof(numMerges)); 

    for (const auto& pair: mergeHistory){
        outfile.write(reinterpret_cast<const char*>(&pair), sizeof(pair)); 
    }

    outfile.close(); 

}

void BPE::load(const std::string& filename){ 
    std::ifstream infile(filename, std::ios::binary); 

    uint32_t magic; 
    infile.read(reinterpret_cast<char*>(&magic), sizeof(magic)); 
    if (magic != magicNumber){
        throw std::runtime_error("Invalid file, mergeRules identifier not found");
    }

    size_t numMerges; 
    infile.read(reinterpret_cast<char*>(&numMerges), sizeof(numMerges)); 

    mergeHistory.clear(); 
    mergeRules.clear(); 
    vocab.clear(); 
    initVocab(); 

    for (size_t i{0}; i < numMerges; ++i){
        std::pair<uint32_t, uint32_t> rule; 
        infile.read(reinterpret_cast<char*>(&rule), sizeof(rule)); 

        uint64_t pairID = (static_cast<uint64_t>(rule.first) << 32 | rule.second);

        updateVocab(pairID); 

    }

    infile.close();
}

PYBIND11_MODULE(bpEncode, m) {
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
































