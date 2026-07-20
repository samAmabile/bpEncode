#include "bpEncode/bpe.hpp"
#include <string>
#include <iostream>
#include <cstdint>
#include <vector>

/******************************************
 *    Byte-Pair Encoding demo program     *
 *    ^^^^^^^^^ ^^^^^^^^ ^^^^ ^^^^^^^     *
 *    -To show how BPE breaks words down   *
 *    words into numeric sub-word tokens  *
 *    -Shows both what those tokens     *
 *    look like in numeric bytes, and     *
 *    what the bytes represent in text    *
 *                                        *
 * For example:                           *
 *    "Encoding codes is the issue        *
 *     03 256 8 14 258 19 45 223 45 187   *
 *    en cod ing cod es is the is sue     *
 *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^* 
 ******************************************/ 



int main(){

    BPE tokenizer; 

    std::string prompt = "Enter text to be tokenized, or enter 'x' to exit";

    while (true){
        
        std::string input; 

        std::cout << prompt << std::endl;
        std::cin >> input; 

        if (input == "x" || input == "X") break;

        const std::string constInput = input; 

        byte_map vocab = tokenizer.fitWords(constInput); 

        std::vector<uint32_t> tokens = tokenizer.tokenize(constInput);
        
        std::vector<std::string> decoded_tokens = {};

        std::cout << "Encoded tokens: " <<std::endl;
        for (const auto& token : tokens){
            std::cout << token << " "; 
            std::vector<uint32_t> temp(1, token);
            std::string decoded_token = tokenizer.decode(temp);
            decoded_tokens.push_back(decoded_token);
        }
        std::cout << std::endl;
  
        std::cout << "Tokens as text: " << std::endl;
        for (const auto& dt : decoded_tokens){

            std::cout << dt << " ";
        }
        std::cout << std::endl;
    }


    return 0;

}
