//
// Created by macross on 3/29/23.
//

#ifndef FLARE_TOKENIZER_HPP
#define FLARE_TOKENIZER_HPP

#include <unordered_map>
#include <sstream>

namespace fl
{

class Tokenizer
{
public:
    Tokenizer(int num_words, int output_len)
            : num_words(num_words), output_len(output_len)
    {
    }


    void Add(const std::vector<std::string> &texts)
    {
        for (auto &sentence: texts) {
            this->Add(sentence);
        }
    }


    void Add(const std::string &sentence)
    {
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
                // word exists, increment its frequency
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
        // skip 0 (for padding) and 1 (not in word index)
        for (int i = 0; i < sorter.size(); i++) {
            this->word_index.insert({sorter[i].second, i + 2});
        }
    }


    Tensor<2> Sequence(const std::vector<std::string> &sentences) const
    {
        Tensor<2> minibatch(static_cast<Eigen::Index>(sentences.size()),
                            this->output_len);
        minibatch.setZero();

        for (int batch = 0; batch < sentences.size(); batch++) {
            std::string words(sentences[batch]);

            // remove all non-alphanumeric characters
            std::transform(words.begin(), words.end(), words.begin(), [](char &c) {
                return std::tolower(c);
            });

            std::istringstream parse(words);
            std::string word;
            int i = 0;

            while (parse >> word && i < output_len) {
                word.erase(std::remove_if(word.begin(), word.end(), [](char &c) {
                    return !std::isalnum(c) && c != ' ';
                }), word.end());

                auto it = this->word_index.find(word);
                minibatch(batch, i) =
                        it != this->word_index.end() && it->second < this->num_words
                        ? it->second : 0;

                i++;
            }
        }

        return minibatch;
    }


    void PrintCounts()
    {
        for (auto &i: this->word_counts) {
            std::cout << i.first << ": " << i.second << "\n";
        }
    }


    void PrintIndices()
    {
        for (auto &i: this->word_index) {
            std::cout << i.first << ": " << i.second << "\n";
        }
    }


    auto Size()
    {
        return this->word_index.size();
    }


private:
    // hold word frequency
    std::unordered_map<std::string, int> word_counts;

    // hold word index, assigned by sorted frequency with 1 being most frequent
    std::unordered_map<std::string, int> word_index;

    const int num_words;
    const int output_len;
};

}

#endif //FLARE_TOKENIZER_HPP
