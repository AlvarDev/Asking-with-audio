from typing import List
from IPython.display import display, HTML

from vertexai.generative_models import GenerativeModel, Part

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import speech

import vertexai

# Define project information
PROJECT_ID = "[YOUR-PROJECT-ID]"  # @param {type:"string"}
ENGINE_ID = "[YOUR-ENGINE-ID]" # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define helper functions
def display_search_response(
    response, # TODO: define from type
    display_citations: bool = False,
    ):

  citations   = response.summary.summary_with_metadata.citation_metadata.citations
  references  = response.summary.summary_with_metadata.references
  summary     = response.summary.summary_with_metadata.summary
  references_display = ""

  if display_citations:
    print(citations)
    print("*" * 100)

  for reference in references:
    references_display += f"""
      > {reference.title}
      {reference.document}
    """

  answer_display = f"""
    "{summary}"
     {references_display}
  """

  # display(HTML(answer_display))
  display(answer_display)
  answer_display = ""


  for result in response.results:
    id     = result.document.id
    title  = result.document.derived_struct_data["title"]
    link   = result.document.derived_struct_data["link"]
    answer = result.document.derived_struct_data["extractive_answers"][0]

    answer_display += f"""
      > {title} -> ({id})
      [Page: {answer["pageNumber"]}]
      {link}

      {answer["content"]}
    """

  # display(HTML(answer_display))
  display("*" * 100) 
  display("Documents found:")
  display("*" * 100)
  display(answer_display)


# Define Search function
def search_sample(
    location: str = "global",
    prompt: str = None,
    search_query: str = "",
    model_version: str = "stable",
) -> List[discoveryengine.SearchResponse]:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{PROJECT_ID}/locations/{location}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_config"

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=1
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version=model_version,
            ),
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble="""Given the conversation between a user and a helpful assistant and some search results,
                            create a final answer for the assistant. The answer should use all relevant information
                            from the search results, not introduce any additional information, and use exactly the
                            same words as the search results when possible. The assistant's answer should be
                            no more than 20 sentences. The assistant's answer should be formatted as a bulleted list.
                            Each list item should start with the "-" symbol.""" if prompt is None else prompt
            ),
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    return response

# Getting Audio text
print("Working...\n")

# Using Gemini
multimodal_model = GenerativeModel("gemini-1.5-pro-preview-0409")

audio_file_uri = ("gs://genai_poc_s/gerar_mapa.ogg")
audio_file = Part.from_uri(audio_file_uri, mime_type="audio/ogg")

prompt = """
Do a Speech-to-Text of the audio, 
do not try to correct the words
"""

contents = [
  audio_file,
  prompt,
]

response = multimodal_model.generate_content(contents)
print("Audio text (Gemini)")
print(response.text)


# Using Speech to Text
speech_client = speech.SpeechClient()
# The name of the audio file to transcribe
gcs_uri = "gs://genai_poc_s/gerar_mapa.ogg"

audio = speech.RecognitionAudio(uri=gcs_uri)

config = speech.RecognitionConfig(
  encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
  sample_rate_hertz=16000,
  language_code="pt-BR",
)

# Detects speech in the audio file
response = speech_client.recognize(config=config, audio=audio)

print("Audio text (STT)")
for result in response.results:
  print(f"Transcript: {result.alternatives[0].transcript}")

print("")
print("=" * 50)
print("In this case we are going to use STT result\n")

search_query = response.results[0].alternatives[0].transcript

print("Asking:")
print(search_query, "\n")

response = search_sample(search_query = search_query)

print("Rsponse:")
display_search_response(response)

