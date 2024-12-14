# Data-Processing-Final-Project


Here we describe the development of each step of the project. The main objective is to apply Natural Language Processing (NLP) to process as set of recipies in the dataset "full_format_recipes". 

I. Analysis of input variables


II.. Implementation of a pipeline for text processing


III. Vector representation of the document usinf three different procedures
 In this section three vectorization methods have been used, TF-IDF, Word2Vec and Trasformers.
 It is important to mention that the input data used here corresponds to the column "Descriptions".
  
   --> TF - IDF
       To develop this method, BoW (Bag of words) has been previously done. BoW provides what we call corpus, which is the base of our TF-IDF model.
       The data used here has been previously treated using the preprocessed function mentioned above. We have used the Gensim libary thanks to which 
       we are able to transform the corpus in to a weighted representation based on the frequency of words in a recipe and across the whole dataset.
     
   --> Word2vec
       For Word2vec vectorization we use Gensim library which generates word embeddings using the previous tokenized corpus. Each description is shown as
       the mean of the embeddings of it's words. Words that are not in the vocubulary of the model are not considered. Results are saved in a numpy array.
       
   --> Transformers
       This is a much more complex method which produces the embeddings according to the context of each word. RobertaModel is used for this purpose. Texts
       are tokenized ensuring a maximum input leght of 512 tokens. Then data is processed in batches of size 16 for more efficient computation. We obtain
       the mean embeddings from the last hidden layer, obtaining one vector representation per input, which is stored in a numpy array. The number of hidden
       layers and attention heads used is set by the roberta-based model. 


1V. Training and evaluation of regression model.
   
    Once the vectorization of the descriptions has been done we proceed to use the embeddings to train and evaluate predicction models. Random Forest and Neural Networks have been used,
    from the scikit learn tool. The results are represented in the table below. Here the two models have been used with they default parameters. Nevertheless, in order to achieve
    better results, GridSearchCV has been tried for Random Forest to perform Hyperparameter selection, with results that do not differ much from the ones obtain with the default parameters.

![image](https://github.com/user-attachments/assets/97f8cf61-5410-4b86-a7d9-a26c3caa4885)

  In this table can clearly be seen how the results are not as expected. R2 score and Mean Square Error (mse) have been chosen to measure the ability of the models according the different input
  vecotrization methods. Focussing on the well known mse, we can appreciate that the best result is obtained for Random forest with TF-IDF while the worst is the Neural Network with the same
  TF-IDF vector.

V. Fine-tunning

   As a final attempt to improve the results, a transformer with a regression head has been implemented here. As a first step we need to traing the model. To do so, we split out dataset into
   two train and test smaller datasets. Then we tokenized the training data, convert it to tensors, create a tensorDataset and use it to train the model using 3 epochs (Three complete
   passes through the entire dataset. As a last step we prepare the test data the same way and we use it to test the predicting performance of the model. The results are shown below.

![image](https://github.com/user-attachments/assets/11efd02e-a7f1-4e57-8bee-63e66d9aa8f5)

