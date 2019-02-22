# Parallel_association_rules
Parallel computing implementation of association rules learning.

he external dependencies:
 * [Numpy](http://www.numpy.org/)
 * [pandas](https://pandas.pydata.org/) 
 * [mlxtend](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association-rules-generation-from-frequent-itemsets) (Only for calculating frequent patterns with "apriori" module. Another alternative is pyspark mllib for frequent pattern tree)



# example:
```
frequent_petterns = mlxtend.frequent_patterns.apriori(
                                                     transaction_DataFrame,
                                                     min_support=0.5,
                                                     max_len=4
                                                     )
                                                     
#  similar to mlxtend.frequent_patterns.association_rules:                                                    
association_rules_results = parallel_association_rules (
                                                        frequent_petterns,
                                                        n_parallel_branch=mp.cpu_count(),
                                                        metric="confidence",
                                                        min_threshold=0.7
                                                        )
```
