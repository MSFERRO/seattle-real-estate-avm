# Dicionário de Dados

Imóveis residenciais da região de Seattle (King County, EUA). Junção entre os arquivos pela coluna `zipcode`.

## `kc_house_data.csv`

| Coluna | Descrição |
|---|---|
| `id` | Identificador do imóvel |
| `date` | Data da venda |
| `price` | Preço de venda em US$ (variável alvo) |
| `bedrooms` | Número de quartos |
| `bathrooms` | Número de banheiros (frações indicam lavabos) |
| `sqft_living` | Área construída habitável, em pés² |
| `sqft_lot` | Área do terreno, em pés² |
| `floors` | Número de andares |
| `waterfront` | Imóvel de frente para a água (0/1) |
| `view` | Qualidade da vista, de 0 a 4 |
| `condition` | Estado de conservação, de 1 a 5 |
| `grade` | Padrão construtivo e de acabamento, de 1 a 13 |
| `sqft_above` | Área construída acima do nível do solo, em pés² |
| `sqft_basement` | Área de porão, em pés² |
| `yr_built` | Ano de construção |
| `yr_renovated` | Ano da última reforma |
| `zipcode` | Código postal |
| `lat` | Latitude |
| `long` | Longitude |
| `sqft_living15` | Área construída média dos 15 imóveis vizinhos mais próximos, em pés² |
| `sqft_lot15` | Área de terreno média dos 15 imóveis vizinhos mais próximos, em pés² |

## `zipcode_demographics.csv`

Dados censitários agregados por código postal. Sufixo `_qty` indica contagem de pessoas, `per_` indica percentual e `_amt` indica valor em US$.

| Coluna | Descrição |
|---|---|
| `ppltn_qty` | População total |
| `urbn_ppltn_qty` | População em área urbana |
| `sbrbn_ppltn_qty` | População em área suburbana |
| `farm_ppltn_qty` | População em área rural agrícola |
| `non_farm_qty` | População em área rural não agrícola |
| `per_urbn` | Percentual da população em área urbana |
| `per_sbrbn` | Percentual da população em área suburbana |
| `per_farm` | Percentual da população em área rural agrícola |
| `per_non_farm` | Percentual da população em área rural não agrícola |
| `medn_hshld_incm_amt` | Renda mediana domiciliar anual |
| `medn_incm_per_prsn_amt` | Renda mediana por pessoa, anual |
| `hous_val_amt` | Valor mediano dos imóveis |
| `edctn_less_than_9_qty` | Pessoas com menos de 9 anos de estudo |
| `edctn_9_12_qty` | Pessoas com 9 a 12 anos de estudo, sem ensino médio completo |
| `edctn_high_schl_qty` | Pessoas com ensino médio completo |
| `edctn_some_clg_qty` | Pessoas com ensino superior incompleto |
| `edctn_assoc_dgre_qty` | Pessoas com curso superior de curta duração (associate degree) |
| `edctn_bchlr_dgre_qty` | Pessoas com bacharelado |
| `edctn_prfsnl_qty` | Pessoas com pós-graduação ou diploma profissional |
| `per_less_than_9` | Percentual com menos de 9 anos de estudo |
| `per_9_to_12` | Percentual com 9 a 12 anos de estudo |
| `per_hsd` | Percentual com ensino médio completo |
| `per_some_clg` | Percentual com ensino superior incompleto |
| `per_assoc` | Percentual com curso superior de curta duração |
| `per_bchlr` | Percentual com bacharelado |
| `per_prfsnl` | Percentual com pós-graduação ou diploma profissional |
| `zipcode` | Código postal |

## `future_unseen_examples.csv`

Mesmas colunas físicas de `kc_house_data.csv`, sem `id`, `date` e `price`. São os imóveis para os quais o preço deve ser previsto.
