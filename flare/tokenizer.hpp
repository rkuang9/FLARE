//
// Created by macross on 3/29/23.
//

#ifndef FLARE_TOKENIZER_HPP
#define FLARE_TOKENIZER_HPP

#include <map>
#include <sstream>

namespace fl
{

class Tokenizer
{
public:
    void Fit(const std::vector<std::string> &texts) {
        for (auto &sentence: texts) {
            this->Fit(sentence);
        }
    }

    void Fit(const std::string &sentence) {
        std::string words(sentence);

        // remove all non-alphanumeric characters
        std::transform(words.begin(), words.end(), words.begin(), [](char &c) {
           return std::tolower(c);
        });

        std::istringstream parse(words);
        std::string word;

        while (parse >> word) {
            word.erase(std::remove_if(word.begin(), word.end(), [](char &c) {
                return !std::isalnum(c) && c != ' ';
            }), word.end());

            auto it = this->word_counts.find(word);

            if (it != this->word_counts.end()) {
                it->second++;
            }
            else {
                this->word_counts.insert({word, 1});
            }
        }
    }


    void Compile()
    {
        std::vector<std::pair<int, std::string>> sorter;
        sorter.reserve(this->word_counts.size());

        // dump word_counts into a vector for sorting
        for (auto &word: this->word_counts) {
            sorter.emplace_back(word.second, word.first);
        }

        // sort by highest to lowest frequency
        std::sort(sorter.rbegin(), sorter.rend());

        // populate the word_index with the word and its vector index
        for (int i = 0; i < sorter.size(); i++) {
            this->word_index.insert({sorter[i].second, i + 1});
        }
    }


    void Print()
    {
        for (auto &i: this->word_counts) {
            std::cout << i.first << ": " << i.second << "\n";
        }
        std::cout << "\nword index\n";

        for (auto &i: this->word_index) {
            std::cout << i.first << ": " << i.second << "\n";
        }

    }

private:
    // hold word frequency
    std::map<std::string, int> word_counts;

    // hold word index, assigned by sorted frequency with 1 being most frequent
    std::map<std::string, int> word_index;
};

}

#endif //FLARE_TOKENIZER_HPP
