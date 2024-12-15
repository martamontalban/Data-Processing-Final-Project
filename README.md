# Data-Processing-Final-Project


Here we describe the development of each step of the project. The main objective is to apply Natural Language Processing (NLP) to process as set of recipies in the dataset "full_format_recipes". 

1. ANALYSIS OF INPUT VARIABLES
First of all, our first step was to import the corresponding libraries and modules needed for our Python program. Those are the pandas library in order to work with structured data, NumPy library for numerical computations and working with arrays, the termcolor library
used to coloriza text printed in the terminal, the reaborn library for creating statistical visualizations, Matplotlib library for creating visualizations in Pyhton, and more libraries that will be mention along the explication of our project and code.

Secondly, we had import the database mention above and we had analyse the input data, which is a diccionary of 20130 cook recipes. Each recipe is define with different variables, in total 8 variables that describe each recipe. The input variables that describe each
recipe are 'directions', 'fat', 'date', 'categories', 'calories', 'description', 'protein', 'rating', 'title', 'ingredients' and 'sodium'. The variables 'fat', 'date', 'calories', 'protein', 'rating' and 'sodium' contain numerical values and the rest of the variables
contain text.

Following, we eliminated null lines in dataset with the 'isna' function. In the code we visualize the relationship between the input variables, we start visualizing the variable 'categories' in order to see the relation of the categories of the different recipes. Some
examples are counting the most common categorie with the 'value_counts()' and also the function 'explode()' is a key in the code because it is really useful to classify all the different words that compose a category. Other example is the creation of a category-rating
relationship, where we plotted a figure evaluating the recipes that have categories in commom. Moreover, we analyze the correlation between numerical elements such as fat, calories... and rating, we computed a correlation matrix where a positive correlation closer to 1
between 2 variables means that the 2 vaariables have a proportional relation, if one increases, the other one increases as well. Correlation close to 0 means that there is no linear relation. And negative correlation means that the variables are inversely related. 

![image](https://github.com/user-attachments/assets/70e7ed09-cbdd-4200-be20-815723f3652c)

2. IMPLEMENTATION OF A PIPELINE FOR TEXT PROCESSING
In this section we first import the NLTK library, which is a powerful toolkit for working with text data in Python. We import the re module, used for pattern matching and text cleaning.

We use preprocessing functions in order to improve the quality of textual data by: eliminating elements that do not add value, unifying the format of the text, reducing the complexity of the vocabulary, in order to prepare the text for models to process it more
efficiently and accurately.

2.1. Preprocess "Desc" column
Here, we analyse the variables description. Firstly, we tokenize all the words that describe the variables desc in order to visualize and analyse them. We conclude that the dictionary contains 5764 terms, where terms refers to all the relative words contained in the
description variable  of all the recipes in the dictionary. 

![image](https://github.com/user-attachments/assets/a42e5239-b445-42b4-bd42-95619402516a) 


2.2. Identify the number of descriptions in which each token appears
Where as the title says, in this step we identify the frequency of each token (words that contains the description of the recipes) in the descriptions appears. In instance, how many times a token appears in different descriptions. We have program the code to do so
following the next goals:
- Build a DataFrame with each token and its frequency in the descriptions.
- Sort tokens by frequency.
- Count and filter the tokens that are infrequent (appear only in one description).
- Visualize the logarithmic distribution of the frequencies with a histogram.
  Printing the following:
  ![image](https://github.com/user-attachments/assets/7b902a8b-48ca-42da-aaa6-ac0df5e3687a)


2.3. Bag of words representation of the corpus
A bag of words is a simple way to convert text into numerical data. We create a vocabulary of unique words (mycorpus) from the entire corpus (all descriptions of the recipes). For each description, count how many times each word containing 'desc' from the vocabulary
appears. The code programmed to do so, returns a sparse vector, which is a list of tuples (token_id, frequency). Where token_id is the ID of the word in the dictionary D and frequency is how many times that word appears in the recipe. 

3. VECTOR REPRESENTATION OF THE DOCUMENTS USING THREE DIFFERENT PROCEDURES
 In this section three vectorization methods have been used, TF-IDF, Word2Vec and Trasformers.
 It is important to mention that the input data used here corresponds to the column "Descriptions".
  
   --> TF - IDF

       To develop this method, BoW (Bag of words) has been previously done. BoW provides what we call corpus, which is the base of our TF-IDF model. The data used here
       has been previously treated using the preprocessed function mentioned above. We have used the Gensim libary thanks to which we are able to transform the corpus
       in to a weighted representation based on the frequency of words in a recipe and across the whole dataset.
       Furthermore, we train a TF-IDF model that uses the corpus (in BoW format) to compute the relative word importances in each recipe. After, we transform the corpus
       converting each document to its TF-IDF representation for further analysis.
     
   --> Word2vec

       For Word2vec vectorization we use Gensim library which generates word embeddings using the previous tokenized corpus. Each description is shown as the mean of the
       embeddings of it's words. Words that are not in the vocubulary of the model are not considered. Results are saved in a numpy array.
       Furthermore, in this section our code uses Word2Vec to represent textual data (descriptions) in a meaningful numerical format. It then computes the average embedding
       for each description in mycorpus to represent the entire description as a single vector.
       
   --> Transformers

       This is a much more complex method which produces the embeddings according to the context of each word. RobertaModel is used for this purpose. Texts are tokenized
       ensuring a maximum input leght of 512 tokens. Then data is processed in batches of size 16 for more efficient computation. We obtain the mean embeddings from the
       last hidden layer, obtaining one vector representation per input, which is stored in a numpy array. The number of hidden layers and attention heads used is set by
       the roberta-based model.
       --> 1. Importing necessary tools
           - datasets: A Hugging Face library to load and process datasets.
           - torch: PyTorch, used for handling tensors and running models like RoBERTa.
       --> 2. Transformer configuration
            In this step, the purpose is to extract semantic embeddings from the recipe's descriptions using a pre-trained transformer model (RoBERTa). These embeddings
            capture the meaning of the descriptions in a numerical form that machine learning models can use.
       --> 3. Apply to our data
       --> 4. Use of PCA for dimensionality reduction
       --> 5. Load saved embeddings


5. TRAINING AND EVALUATION OF REGRESSION MODELS.
   
    Once the vectorization of the descriptions has been done we proceed to use the embeddings to train and evaluate predicction models. Random Forest and Neural Networks have been used,
    from the scikit learn tool. The results are represented in the table below. Here the two models have been used with they default parameters. Nevertheless, in order to achieve
    better results, GridSearchCV has been tried for Random Forest to perform Hyperparameter selection, with results that do not differ much from the ones obtain with the default parameters.

![image](https://github.com/user-attachments/assets/97f8cf61-5410-4b86-a7d9-a26c3caa4885)

  In this table can clearly be seen how the results are not as expected. R2 score and Mean Square Error (mse) have been chosen to measure the ability of the models according the different input
  vecotrization methods. Focussing on the well known mse, we can appreciate that the best result is obtained for Random forest with TF-IDF while the worst is the Neural Network with the same
  TF-IDF vector.

5. FINE-TUNNING.

   As a final attempt to improve the results, a transformer with a regression head has been implemented here. As a first step we need to traing the model. To do so, we split out dataset into
   two train and test smaller datasets. Then we tokenized the training data, convert it to tensors, create a tensorDataset and use it to train the model using 3 epochs (Three complete
   passes through the entire dataset. As a last step we prepare the test data the same way and we use it to test the predicting performance of the model. The results are shown below.

![image](https://github.com/user-attachments/assets/11efd02e-a7f1-4e57-8bee-63e66d9aa8f5)

