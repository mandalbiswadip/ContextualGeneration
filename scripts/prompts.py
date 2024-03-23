from typing import List
class CitationTextGenerationConfig:
    def __init__(self):
        self.Dominant_only = {
            "Zero_Shot": """
            We are writing the related work section of our paper. We have an incomplete section where one we want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, complete it by putting summarizing the cited papers. The position where the summarized text goes are indicated by `[Dominant]`. We have also been given a list of cited papers(Citatation Mark, Title, Abstract). The cited paper can be of two types: 
            1. Dominant (These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.)
            2. These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations

            Output format: Return only the piece of text required to complete the related work section. Return it in string format

            Incomplete Related work section:

            """,

            "Few_Shots": """ We are writing the related work section of our paper. We have an incomplete section where one we want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, complete it by summarizing the cited papers in a short piece of text. The position where the summarized text goes are indicated by [Dominant]. We have also been given a list of cited papers(Citatation Mark, Title, Abstract). The cited paper can be of two types:
            Dominant: These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.
            Reference: These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations
            Output format: Return only the piece of text required to complete the related work section. Return it in string format

            Incomplete Related work section: 
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.\n [Dominant] \nMore recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.\nIn their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.\nTo incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.\nBecause we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            List of cited papers:

            Citation mark: Zhang and Clark (2008) 

            Citation Type: Dominant 

            Title: Joint Word Segmentation and POS Tagging Using a Single Perceptron

            Abstract: For Chinese POS tagging, word segmentation is a preliminary step. To avoid error propagation and improve segmentation by utilizing POS information, segmentation and tagging can be performed simultaneously. A challenge for this joint approach is the large combined search space, which makes efficient decoding very hard. Recent research has explored the integration of segmentation and POS tagging, by decoding under restricted versions of the full combined search space. In this paper, we propose a joint segmentation and POS tagging model that does not impose any hard constraints on the interaction between word and POS information. Fast decoding is achieved by using a novel multiple-beam search algorithm. The system uses a discriminative statistical model, trained using the generalized perceptron algorithm. The joint model gives an error reduction in segmentation accuracy of 14.6% and an error reduction in tagging accuracy of 12.2%, compared to the traditional pipeline approach.

            Output:
            Zhang and Clark (2008) proposed an incremental joint segmentation and POS tagging model, with an effective feature set for Chinese.\nHowever, it requires to computationally expensive multiple beams to compare words of different lengths using beam search.


            Incomplete Related work section: 
            """,

            "Two_Shots": """We are writing the related work section of our paper. We have an incomplete section where one we want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, complete it by summarizing the cited papers in a short piece of text. The position where the summarized text goes are indicated by [Dominant]. We have also been given a list of cited papers(Citatation Mark, Title, Abstract). The cited paper can be of two types:
            Dominant: These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.
            Reference: These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations
            Output format: Return only the piece of text required to complete the related work section. Return it in string format

            Incomplete Related work section: 
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.\n [Dominant] \nMore recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.\nIn their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.\nTo incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.\nBecause we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            List of cited papers:

            Citation mark: Zhang and Clark (2008) 

            Citation Type: Dominant 

            Title: Joint Word Segmentation and POS Tagging Using a Single Perceptron

            Abstract: For Chinese POS tagging, word segmentation is a preliminary step. To avoid error propagation and improve segmentation by utilizing POS information, segmentation and tagging can be performed simultaneously. A challenge for this joint approach is the large combined search space, which makes efficient decoding very hard. Recent research has explored the integration of segmentation and POS tagging, by decoding under restricted versions of the full combined search space. In this paper, we propose a joint segmentation and POS tagging model that does not impose any hard constraints on the interaction between word and POS information. Fast decoding is achieved by using a novel multiple-beam search algorithm. The system uses a discriminative statistical model, trained using the generalized perceptron algorithm. The joint model gives an error reduction in segmentation accuracy of 14.6% and an error reduction in tagging accuracy of 12.2%, compared to the traditional pipeline approach.

            Output:
            Zhang and Clark (2008) proposed an incremental joint segmentation and POS tagging model, with an effective feature set for Chinese.\nHowever, it requires to computationally expensive multiple beams to compare words of different lengths using beam search.

            Incomplete Related work section: 
            There are also existing works that exploited neural networks to learn translation probabilities for translation rules used in the phrase-based translation model.\nNamely, these methods estimated translation probabilities for phrase pairs extracted from the parallel corpus.\n [Dominant] \nGao et al. (2014) and Zhang et al. (2014) proposed methods to learn continuous space phrase representations and use the similarity between the source and target phrases as translation probabilities for phrase pairs.\nAll these three methods can only be used for the phrase-based translation model, not for syntaxbased translation models.


            List of cited papers:

            Citation mark: Schwenk (2012)

            Citation Type: Dominant 

            Title: Continuous Space Translation Models for Phrase-Based Statistical Machine Translation

            Abstract: This paper presents a new approach to perform the estimation of the translation model probabilities of a phrase-based statistical machine translation system. We use neural networks to directly learn the translation probability of phrase pairs using continuous representations. The system can be easily trained on the same data used to build standard phrase-based systems. We provide experimental evidence that the approach seems to be able to infer meaningful translation probabilities for phrase pairs not seen in the training data, or even predict a list of the most likely translations given a source phrase. The approach can be used to rescore n-best lists, but we also discuss an integration into the Moses decoder. A preliminary evaluation on the English/French IWSLT task achieved improvements in the BLEU score and a human analysis showed that the new model often chooses semantically better translations. Several extensions of this work are discussed.

            Output:
            Schwenk (2012) proposed a continuous space translation model, which calculated the translation probability for each word in the target phrase and then multiplied the probabilities together as the translation probability of the phrase pair.


            Incomplete Related work section: """,

            "Two_shots_infilling" : """We are writing the related work section of our paper. We have an incomplete section where one we want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, complete it by summarizing the cited papers in a short piece of text. The position where the summarized text goes are indicated by [Dominant]. We have also been given a list of cited papers(Citatation Mark, Title, Abstract). The cited paper can be of two types:
            Dominant: These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.
            Reference: These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations
            Output format: Return the complete related work section. Return it in string format. the cited paper summary should be seperated from the rest of the related work section by <sep> and </sep>. The return format is <Related work section before [Dominant]> <sep> cited paper summary </sep> <Related work section after [Dominant]>

            Incomplete Related work section: 
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.\n [Dominant] \nMore recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.\nIn their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.\nTo incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.\nBecause we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            List of cited papers:

            Citation mark: Zhang and Clark (2008) 

            Citation Type: Dominant 

            Title: Joint Word Segmentation and POS Tagging Using a Single Perceptron

            Abstract: For Chinese POS tagging, word segmentation is a preliminary step. To avoid error propagation and improve segmentation by utilizing POS information, segmentation and tagging can be performed simultaneously. A challenge for this joint approach is the large combined search space, which makes efficient decoding very hard. Recent research has explored the integration of segmentation and POS tagging, by decoding under restricted versions of the full combined search space. In this paper, we propose a joint segmentation and POS tagging model that does not impose any hard constraints on the interaction between word and POS information. Fast decoding is achieved by using a novel multiple-beam search algorithm. The system uses a discriminative statistical model, trained using the generalized perceptron algorithm. The joint model gives an error reduction in segmentation accuracy of 14.6% and an error reduction in tagging accuracy of 12.2%, compared to the traditional pipeline approach.

            Output:
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.\n <sep>            Zhang and Clark (2008) proposed an incremental joint segmentation and POS tagging model, with an effective feature set for Chinese.\nHowever, it requires to computationally expensive multiple beams to compare words of different lengths using beam search. </sep> \nMore recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.\nIn their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.\nTo incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.\nBecause we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            Incomplete Related work section: 
            There are also existing works that exploited neural networks to learn translation probabilities for translation rules used in the phrase-based translation model.\nNamely, these methods estimated translation probabilities for phrase pairs extracted from the parallel corpus.\n [Dominant] \nGao et al. (2014) and Zhang et al. (2014) proposed methods to learn continuous space phrase representations and use the similarity between the source and target phrases as translation probabilities for phrase pairs.\nAll these three methods can only be used for the phrase-based translation model, not for syntaxbased translation models.


            List of cited papers:

            Citation mark: Schwenk (2012)

            Citation Type: Dominant 

            Title: Continuous Space Translation Models for Phrase-Based Statistical Machine Translation

            Abstract: This paper presents a new approach to perform the estimation of the translation model probabilities of a phrase-based statistical machine translation system. We use neural networks to directly learn the translation probability of phrase pairs using continuous representations. The system can be easily trained on the same data used to build standard phrase-based systems. We provide experimental evidence that the approach seems to be able to infer meaningful translation probabilities for phrase pairs not seen in the training data, or even predict a list of the most likely translations given a source phrase. The approach can be used to rescore n-best lists, but we also discuss an integration into the Moses decoder. A preliminary evaluation on the English/French IWSLT task achieved improvements in the BLEU score and a human analysis showed that the new model often chooses semantically better translations. Several extensions of this work are discussed.

            Output:
			There are also existing works that exploited neural networks to learn translation probabilities for translation rules used in the phrase-based translation model.\nNamely, these methods estimated translation probabilities for phrase pairs extracted from the parallel corpus.\n <sep> Schwenk (2012) proposed a continuous space translation model, which calculated the translation probability for each word in the target phrase and then multiplied the probabilities together as the translation probability of the phrase pair. </sep> \nGao et al. (2014) and Zhang et al. (2014) proposed methods to learn continuous space phrase representations and use the similarity between the source and target phrases as translation probabilities for phrase pairs.\nAll these three methods can only be used for the phrase-based translation model, not for syntaxbased translation models.



            Incomplete Related work section: """
            ,

            "Few_Shots_Length_Control" : """We are writing the related work section of our paper. We have an incomplete section where one we want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, complete it by summarizing the cited papers in a short piece of text given the Number of words in output>. The position where the summarized text goes are indicated by [Dominant]. We have also been given a list of cited papers(Citatation Mark, Title, Abstract). The cited paper can be of two types:
            Dominant: These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.
            Reference: These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations
            Output format: Return only the piece of text required to complete the related work section. Return it in string format. The number of words in the output should be strictly the same as <Number of words in output>

            Incomplete Related work section: 
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.
            [Dominant] 
            More recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.
            In their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.
            To incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.
            Because we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            List of cited papers:

            Citation mark: Zhang and Clark (2008) 

            Citation Type: Dominant 

            Title: Joint Word Segmentation and POS Tagging Using a Single Perceptron

            Abstract: For Chinese POS tagging, word segmentation is a preliminary step. To avoid error propagation and improve segmentation by utilizing POS information, segmentation and tagging can be performed simultaneously. A challenge for this joint approach is the large combined search space, which makes efficient decoding very hard. Recent research has explored the integration of segmentation and POS tagging, by decoding under restricted versions of the full combined search space. In this paper, we propose a joint segmentation and POS tagging model that does not impose any hard constraints on the interaction between word and POS information. Fast decoding is achieved by using a novel multiple-beam search algorithm. The system uses a discriminative statistical model, trained using the generalized perceptron algorithm. The joint model gives an error reduction in segmentation accuracy of 14.6% and an error reduction in tagging accuracy of 12.2%, compared to the traditional pipeline approach.

            <Number of words in output>: 37

            Output:
            ```Zhang and Clark (2008) proposed an incremental joint segmentation and POS tagging model, with an effective feature set for Chinese.
            However, it requires to computationally expensive multiple beams to compare words of different lengths using beam search.```

            Explanation: The generated output has 37 words which starts with the word `Zhang` and ends with the word `search`

            Incomplete Related work section: 
            """,

            "Two_Shots_Length_Control": """We have written the abstract and introduction section of our NLP paper. We have an incomplete related work section where we have discussed most of the cited paper's work except a few. We want a set of cited papers to be summarized in the context of the current related work section. Given the incomplete related work section, list of cited papers and <Number of words in output>, complete it by summarizing the cited papers in the related work section context in a short piece of text. The position where the summarized text goes are indicated by the special token: '[Dominant]'. The cited paper can be of two types reffered to as Citattion Type:
            Dominant: These citations are discussed in detail, usually via summarization of their content, and are often longer than reference citations.
            Reference: These citations are not discussed in detail. Reference citations tend to be more abstractive than dominant citations
            Output format: Summarize the cited papers givem their Citation Mark, Citation Type, Title and Abstract. Return only the piece of text required to complete the related work section. Return it in string format. The number of words in the output should be the same as <Number of words in output>

            Incomplete Related work section: 
            Among the many recent works on joint segmentation and POS tagging for Chinese, the linear-time incremental models by Zhang and Clark (2008) and Zhang and Clark (2010) largely inspired our model.
            [Dominant] 
            More recently, Zhang and Clark (2010) proposed an efficient character-based decoder for their word-based model.
            In their new model, a single beam suffices for decoding; hence, they reported that their model is practically ten times as fast as their original model.
            To incorporate the word-level features into the character-based decoder, the features are decomposed into substring-level features, which are effective for incomplete words to have comparable scores to complete words in the beam.
            Because we found that even an incremental approach with beam search is intractable if we perform the wordbased decoding, we take a character-based approach to produce our joint model.


            List of cited papers:

            1.

            Citation mark: Zhang and Clark (2008) 

            Citation Type: Dominant 

            Title: Joint Word Segmentation and POS Tagging Using a Single Perceptron

            Abstract: For Chinese POS tagging, word segmentation is a preliminary step. To avoid error propagation and improve segmentation by utilizing POS information, segmentation and tagging can be performed simultaneously. A challenge for this joint approach is the large combined search space, which makes efficient decoding very hard. Recent research has explored the integration of segmentation and POS tagging, by decoding under restricted versions of the full combined search space. In this paper, we propose a joint segmentation and POS tagging model that does not impose any hard constraints on the interaction between word and POS information. Fast decoding is achieved by using a novel multiple-beam search algorithm. The system uses a discriminative statistical model, trained using the generalized perceptron algorithm. The joint model gives an error reduction in segmentation accuracy of 14.6% and an error reduction in tagging accuracy of 12.2%, compared to the traditional pipeline approach.

            <Number of words in output>: 37

            Output:
            ```Zhang and Clark (2008) proposed an incremental joint segmentation and POS tagging model, with an effective feature set for Chinese.
            However, it requires to computationally expensive multiple beams to compare words of different lengths using beam search.```

            Explanation: The generated output has 37 words which starts with the word `Zhang` and ends with the word `search`

            Incomplete Related work section: 
            There are also existing works that exploited neural networks to learn translation probabilities for translation rules used in the phrase-based translation model.\nNamely, these methods estimated translation probabilities for phrase pairs extracted from the parallel corpus.\n [Dominant] \nGao et al. (2014) and Zhang et al. (2014) proposed methods to learn continuous space phrase representations and use the similarity between the source and target phrases as translation probabilities for phrase pairs.\nAll these three methods can only be used for the phrase-based translation model, not for syntaxbased translation models.


            List of cited papers:

            1.

            Citation mark: Schwenk (2012)

            Citation Type: Dominant 

            Title: Continuous Space Translation Models for Phrase-Based Statistical Machine Translation

            Abstract: This paper presents a new approach to perform the estimation of the translation model probabilities of a phrase-based statistical machine translation system. We use neural networks to directly learn the translation probability of phrase pairs using continuous representations. The system can be easily trained on the same data used to build standard phrase-based systems. We provide experimental evidence that the approach seems to be able to infer meaningful translation probabilities for phrase pairs not seen in the training data, or even predict a list of the most likely translations given a source phrase. The approach can be used to rescore n-best lists, but we also discuss an integration into the Moses decoder. A preliminary evaluation on the English/French IWSLT task achieved improvements in the BLEU score and a human analysis showed that the new model often chooses semantically better translations. Several extensions of this work are discussed.

            <Number of words in output>: 34

            Output:
            ```Schwenk (2012) proposed a continuous space translation model, which calculated the translation probability for each word in the target phrase and then multiplied the probabilities together as the translation probability of the phrase pair.```


            Explanation: The generated output has 34 words which starts with the word `Schwenk` and ends with the word `pair`

            Incomplete Related work section:"""
        }


    def get_zero_shot_prompt(self, incomplete_related_work:str, cited_papers: List[dict]):
        """get the zero shots prompt given the following

        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = []
        for paper in cited_papers:
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"List of cited papers:\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Zero_Shot"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}"

        return completed_prompt
    

    def get_few_shot_prompt(self, incomplete_related_work:str, cited_papers: List[dict]):
        """get the few shots prompt given the following
        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = []
        for paper in cited_papers:
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"List of cited papers:\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Few_Shots"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}\nOutput:"

        return completed_prompt
    

    def get_two_shot_prompt(self, incomplete_related_work:str, cited_papers: List[dict]):
        """get the few shots prompt given the following
        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = []
        for paper in cited_papers:
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"List of cited papers:\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Two_Shots"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}\nOutput:"

        return completed_prompt
    

    def get_two_shot_prompt_infilling(self, incomplete_related_work:str, cited_papers: List[dict]):
        """get the few shots prompt given the following
        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = []
        for paper in cited_papers:
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"List of cited papers:\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Two_shots_infilling"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}\nOutput:"

        return completed_prompt

    

    def get_few_shot_length_control_prompt(self, incomplete_related_work:str, cited_papers: List[dict], length:int):
        """get the few shots prompt given the following
        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = []
        for paper in cited_papers:
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"List of cited papers:\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Few_Shots_Length_Control"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}\n<Number of words in output>: {length}\n\nOutput:"

        return completed_prompt
    
    def get_two_shot_length_control_prompt(self, incomplete_related_work:str, cited_papers: List[dict], length:int):
        """get the few shots prompt given the following
        Args:
            incomplete_related_work (str): incomplete related work section with either [Dominant] or [Reference] masking
            cited_papers (List[dict]): cited paper details. i.e. citation mark, citation type, cited paper title and abstract

        Returns:
            str: complete prompt for LLM
        """
        # Extract incomplete related work section
        incomplete_related_work_text = incomplete_related_work.strip()

        # Extract cited papers and their details
        cited_paper_texts = ["List of cited papers:\n\n"]
        for i, paper in enumerate(cited_papers):
            citation_mark = paper.get("citation_mark", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            citation_type = paper.get("citation_type", "")
            paper_text = f"{i+1}\n\nCitation mark: {citation_mark}\n\nCitation Type: {citation_type}\n\nTitle: {title}\n\nAbstract: {abstract}\n\n"
            cited_paper_texts.append(paper_text)

        # Combine cited papers into a single string
        cited_papers_text = "\n\n".join(cited_paper_texts)

        # Complete the related work section by summarizing cited papers
        completed_prompt = self.Dominant_only["Two_Shots_Length_Control"] + f"{incomplete_related_work_text}\n\n{cited_papers_text}\n<Number of words in output>: {length}\n\nOutput:"

        return completed_prompt


if __name__ == "__main__":
    # Example usage:
    config = CitationTextGenerationConfig()
    incomplete_related_work = """
    Text simplification has often been addressed as a monolingual translation process, which generates a simplified version of a complex text.\n [Dominant] \nCoster and Kauchak (2011) use a Phrase-Based Machine Translation (PBMT) system with support for deleting phrases, while Wubben et al. (2012) extend a PBMT system with a reranking heuristic (PBMT-R).\nWoodsend and Lapata (2011) propose a model based on a quasisynchronous grammar, a formalism able to capture structural mismatches and complex rewrite operations.\nNarayan and Gardent (2014) combine a sentence splitting and deletion model with PBMT-R.\nThis model has been shown to perform competitively with neural models on automatic metrics, though it is outperformed using human judgments (Zhang and Lapata, 2017) .
    """

    # Example list of cited papers
    cited_papers = [
        {
            "citation_mark": "Zhu et al. (2010)",
            "citation_type": "Dominant",
            "title": "A Monolingual Tree-based Translation Model for Sentence Simplification",
            "abstract": "In this paper, we consider sentence simplification as a special form of translation with the complex sentence as the source and the simple sentence as the target. We propose a Tree-based Simplification Model (TSM), which, to our knowledge, is the first statistical simplification model covering splitting, dropping, reordering and substitution integrally. We also describe an efficient method to train our model with a large-scale parallel dataset obtained from the Wikipedia and Simple Wikipedia. The evaluation shows that our model achieves better readability scores than a set of baseline systems."
        },
        {
            "citation_mark": "Zhu et al. (2010)",
            "citation_type": "Dominant",
            "title": "A Monolingual Tree-based Translation Model for Sentence Simplification",
            "abstract": "In this paper, we consider sentence simplification as a special form of translation with the complex sentence as the source and the simple sentence as the target. We propose a Tree-based Simplification Model (TSM), which, to our knowledge, is the first statistical simplification model covering splitting, dropping, reordering and substitution integrally. We also describe an efficient method to train our model with a large-scale parallel dataset obtained from the Wikipedia and Simple Wikipedia. The evaluation shows that our model achieves better readability scores than a set of baseline systems."
        }
    ]
    completed_prompt = config.get_two_shot_prompt_infilling(incomplete_related_work, cited_papers)
    print(completed_prompt)
