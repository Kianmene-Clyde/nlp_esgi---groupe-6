# Rapport NLP TD6

## Tests avec différentes tailles de chunks

Pour commencer nos tests, nous avons commencé avec différentes tailles de chunks. 

Pour considérer la reply accuracy nous prenons en compte percent_correct plutôt que reply_similarity 
car percent_correct est le ponrcentage des réponses avec une reply_similarity supérieure à 0.7 tandis que reply_similarity est la moyenne des reply similarity entre les réponses attendues et obtenues et peut donc être faussé s'il ya des outlyers.

Nous observons que la taille des chunks est grande plus le score MRR est bon puis le score rechute entre une chunk_size de 512 et 1024.
Néanmoins, les percent_correct sont identiques entre les chunk_size 128 et 1024.


| chunk_size | nb_chunks | mrr     | percent_correct | reply_similarity |
|------------|-----------|---------|-----------------|------------------|
| 128        | 594       | ~0,1761 | ~0.6667         | ~0.6497          |
| 256        | 347       | ~0,1837 | ~0.4444         | ~0.5699          |
| 512        | 220       | ~0,2651 | ~0.4444         | ~0.5544          |
| 1024       | 163       | ~0,1485 | ~0.6667         | ~0.6658          |


Nous pouvons considérer que le meilleur résultat obtenu est celui avec une chunk_size de 128 car a le meilleur MRR et percent_correct.


## Tests avec chunk_size et overlap

La prochaine expérimentation vise à observer si l'on peut obtenir des meilleurs résultats pour les chunk_size testées ci-dessus avec un overlap naïf de ~10%. (on ajoute simplement 10% de ses chunk adjacents à un chunk).

Nous constatons que que le meilleur score obtenu en reply accuracy est encore une fois le plus petit chunck avec un percent correct près 0.9 et qu'en retrieval 
c'est encore une fois le chunk à 512 avec un mrr à près de 0.26.

L'overlap a augmenté le percent correct et  fait chuter le mrr de toutes les chunk sizes mise à part ceux du chunk_size à de 1024

| chunk_size | overlap | nb_chunk | mrr     | percent_correct | reply_similarity |
|------------|---------|----------|---------|-----------------|------------------|
| 128        | 12      | 627      | ~0,1609 | ~0.8889         | ~0.7770          |
| 256        | 25      | 355      | ~0,1808 | ~0.6667         | ~0.6843          |
| 512        | 51      | 222      | ~0,2547 | ~0.5556         | ~0.5980          |
| 1024       | 102     | 164      | ~0.1913 | ~0.5556         | ~0.5923          |


Selon [cet article](https://procogia.com/unlocking-rags-potential-mastering-advanced-techniques-part-1/) les petits chunk size permettent une meilleur similarité entre les questions et les réponses à défaut de perdre du contexte.

Cela pourrait expliquer le faible MRR malgrès meilleur percent_correct un malgrès percent correct à 0.9 pour la chunk size à 128.
Pour y pallier, une solution serait d'implémenter 




