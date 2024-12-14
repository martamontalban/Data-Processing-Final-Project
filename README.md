# Data-Processing-Final-Project


Here we describe the development of each step of the project. The main objective is to apply Natural Language Processing (NLP) to process as set of recipies in the dataset "full_format_recipes". 

1. Analysis of input variables


2. Implementation of a pipeline for text processing


3. Vector representation of the document usinf three different procedures

  In this section three vectorization methods have been used, TF-IDF, Word2Vec and Trasformers.
  It is important to mention that the input data used here corresponds to the column "Descriptions".
   
    --> TF - IDF
        To develop this method, BoW (Bag of words) has been previously developed. BoW provides what we call "corpus" which is the base of our TF-IDF model.
        The data used here has been previously treated using the preprocessed function mentioned above.
      
    --> Word2vec
        For Word2vec vectorization preprocessed data is again used, tokenized elements stored in my corpus. Here we represent the descriptions as the average 
        of worg embeddings.
        
    --> Transformers
        This is a much more complex method which produces the embeddings according to the context of each word. RobertaModel is used for this purpose.


5. Training and evaluation of regression models

  Once the vectorization of the descriptions has been done we proceed to use the embeddings to train and evaluate predicction models. Random Forest and Neural Networks have been used
  from the scikit learn tool. The results are represented in two different graphics. Here the two model have been used with they default parameters. Nevertheless, GridSearchCV has been 
  used for Random Forest to perform Hyperparameter selection, with results that do not differ much from the ones obtain with the default parameters.

  ![image](https://github.com/user-attachments/assets/ead3c266-c380-44fb-be87-649e1d5699db)

  

6. Fine-tunning

   As a final attempt to improve the results, a transformer with a regression head has been implemented here. 
