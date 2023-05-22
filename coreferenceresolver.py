# import spacy
# import spacy_transformers
# import spacy_experimental
# from spacy_experimental import coref
#
# nlp = spacy.load("Dataset/en_core_web_sm/en_core_web_sm-3.5.0")
# nlp_coref = spacy.load("Dataset/en_coreference_web_trf/en_coreference_web_trf-3.4.0a0")
# # use replace_listeners for the coref components
# nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
# nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])
#
# # we won't copy over the span cleaner
# nlp.add_pipe("coref", source=nlp_coref)
# nlp.add_pipe("span_resolver", source=nlp_coref)
#
# doc = nlp("John Smith called from London, he said it's raining in the city.")
#
# print(doc.spans)

# # Print out component names
# print("Pipeline components")
# for i, pipe in enumerate(nlp.pipe_names):
#     print(f"{i}: {pipe}")
#
# # Print out clusters
# print("Found clusters")
# for cluster in doc.spans:
#     print(f"{cluster}: {doc.spans[cluster]}")
#
#
# # Define lightweight function for resolving references in text
# def resolve_references(doc) -> str:
#     # token.idx : token.text
#     token_mention_mapper = {}
#     output_string = ""
#     clusters = [
#         val for key, val in doc.spans.items() if key.startswith("coref_cluster")
#     ]
#
#     # Iterate through every found cluster
#     for cluster in clusters:
#         first_mention = cluster[0]
#         # Iterate through every other span in the cluster
#         for mention_span in list(cluster)[1:]:
#             # Set first_mention as value for the first token in mention_span in the token_mention_mapper
#             token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
#
#             for token in mention_span[1:]:
#                 # Set empty string for all the other tokens in mention_span
#                 token_mention_mapper[token.idx] = ""
#
#     # Iterate through every token in the Doc
#     for token in doc:
#         # Check if token exists in token_mention_mapper
#         if token.idx in token_mention_mapper:
#             output_string += token_mention_mapper[token.idx]
#         # Else add original token text
#         else:
#             output_string += token.text + token.whitespace_
#
#     return output_string
#
#
# print("Document with resolved references = ")
# print(resolve_references(doc))
