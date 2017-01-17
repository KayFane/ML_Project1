# ML_Project1

1. About prediction, I think it's important to identify depending variables, X. For predicting participation, I did a initial experiment on how the participation in previous events indicate the participation in the current event. 

  1.1 In order to do so, I used the participation at the year 2016 as y; and the participation from year 2003 to year 2015 as X. 
  
  1.2 In this training set, I omitted all the entries with no signle appearance between 2003 and 2015. The reason is that when we use the same model to predict one person's participation at the year 2017, this person must have shown up at least once from 2003, or she / he won't be in the date set. 
  
  1.3 I didn't implement any prediction model. I used scikit-learn library, which does not satisfy the project requirement. It's faster to get reliable results, and it's nice to be used as a benchmark for our implementations. 
  
  
2. The report template is 'LaTex' file, which is easy to format and compile. It's also easy to learn. We can also start in Word, and I can format our report in 'Latex'. 

* My code sometimes confuses me, so any comments are welcome (especially about coding convention) 

