import os

from langchain_core.prompts.prompt import PromptTemplate

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


RESTRICT_NER_TEMPLATE: str = """Role: You are an expert in extracting specific types of named entities. 

Task: Please extract the specified types of entities from the input question and return them in JSON format.

Specified Entity Types:
- Person: Specific individual name, e.g., King Edward II, Rihanna
- Organization: Specific organization name, e.g., Cartoonito, Apalachee
- Location: Specific location name, e.g., Fort Richardson, California
- Artwork: Specific artwork (book, song, painting, etc.) name, e.g., Die schweigsame Frau
- Event: Specific event, e.g., Prix Benois de la Danse
- Other propernoun: e.g.,  Cold War, Laban Movement Analysis


Requirements:
- The extracted entities must fall within the specified types of entities.
- If no specified type of entity is present in the question, return an empty list.
- All entities should be in the same level list; do not return them separated by type.
- The returned result must not contain the ' symbol.

Outputs must follow the JSON code format below without additional JSON identifiers, where "named_entities" represents the list of extracted entity names.
```json
{{"named_entities": ["A", "B", ...]}}
```

Examples:
{examples}

Question: {question}
Answer:"""

RESTRICT_NER_PROMPT = PromptTemplate(
    input_variables=["question", "examples"],
    template=RESTRICT_NER_TEMPLATE
)


RESTRICT_NER_EXAMPLE_PROMPT = PromptTemplate.from_template(
    'Question: {question}\nAnswer: {answer}'
)




ENTITY_LINK_REVIEW_TEMPLATE = """Role: You are an expert in entity recognition.

Task: Given multiple pairs of entities, use your expertise and contextual information to judiciously determine whether each pair of entities refers to the same entity.

Requirements:
1. For each pair of entities, extract the following information:
   - source_entity: The first entity in the pair.
   - target_entity: The second entity in the pair.
   - is_same_entity: Whether they refer to the same entity or not; input 'true' if they are, otherwise 'false'.
   - reasoning: Your rationale for the judgment.
   - score: A confidence score as a floating-point number between 0 and 1, where higher scores indicate higher confidence.

   The format for each judgment result is: 
   (<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<is_same_entity>{tuple_delimiter}<reasoning>{tuple_delimiter}<score>)

   **Important**: The output entities must remain consistent with the input entities.


2. Return the judgment results for each pair of entities using a single list, with **{record_delimiter}** as the delimiter between records.


3. After the completion of the task, output {completion_delimiter}.

**Important**: Please be cautious when affirming that a pair of entities is the same. If you are unsure, judge that the pair of entities are not the same.

#############################
-Example-
######################
Example 1:

context: What month did the Tripartite discussions begin between Britain, France, and the country where, despite being headquartered in the nation called the nobilities commonwealth, the top-ranking Warsaw Pact operatives originated?
pairs of entities: (Tripartite discussions,tripartite negotiations), (Tripartite discussions,discussions), (Tripartite discussions,three parts), (France,france), (France,french), (France,frances), (Warsaw Pact,warsaw pact), (Warsaw Pact,creation of warsaw pact), (Warsaw Pact,the treaty), (Britain,britannia), (Britain,the united kingdom), (Britain,united kingdom)
output:
(Tripartite discussions{tuple_delimiter}tripartite negotiations{tuple_delimiter}true{tuple_delimiter}Tripartite discussions and tripartite negotiations refer to the same concept of three-party talks, just using different terms. Therefore, they are the same entity.{tuple_delimiter}0.95){record_delimiter}
(Tripartite discussions{tuple_delimiter}discussions{tuple_delimiter}false{tuple_delimiter}Tripartite discussions specifically refer to discussions involving three parties, while discussions is a general term that can refer to any conversation or debate. Therefore, they are not the same entity.{tuple_delimiter}0.9){record_delimiter}
(Tripartite discussions{tuple_delimiter}three parts{tuple_delimiter}false{tuple_delimiter}Tripartite discussions refer to a type of negotiation or dialogue involving three parties, whereas three parts is a general term that can refer to any division into three sections. Therefore, they are not the same entity.{tuple_delimiter}0.85){record_delimiter}
(France{tuple_delimiter}france{tuple_delimiter}true{tuple_delimiter}France and france are the same entity, with the only difference being the capitalization of the first letter.{tuple_delimiter}0.99){record_delimiter}
(France{tuple_delimiter}french{tuple_delimiter}false{tuple_delimiter}France is a country, while french refers to the language or people from France. Therefore, they are not the same entity.{tuple_delimiter}0.95){record_delimiter}
(France{tuple_delimiter}frances{tuple_delimiter}false{tuple_delimiter}France is a country, while frances is a plural form of a name or term that does not refer to the country. Therefore, they are not the same entity.{tuple_delimiter}0.9){record_delimiter}
(Warsaw Pact{tuple_delimiter}warsaw pact{tuple_delimiter}true{tuple_delimiter}Warsaw Pact and warsaw pact are the same entity, with the only difference being the capitalization of the first letter.{tuple_delimiter}0.99){record_delimiter}
(Warsaw Pact{tuple_delimiter}creation of warsaw pact{tuple_delimiter}false{tuple_delimiter}Warsaw Pact refers to the military alliance itself, while creation of warsaw pact refers to the event or process of its formation. Therefore, they are not the same entity.{tuple_delimiter}0.9){record_delimiter}
(Warsaw Pact{tuple_delimiter}the treaty{tuple_delimiter}false{tuple_delimiter}Warsaw Pact refers to the military alliance, while the treaty is a general term that could refer to any formal agreement. Therefore, they are not the same entity.{tuple_delimiter}0.85){record_delimiter}
(Britain{tuple_delimiter}britannia{tuple_delimiter}false{tuple_delimiter}Britain is a country, while Britannia is a historical and poetic name for Great Britain, often personified. They are related but not the same entity.{tuple_delimiter}0.9){record_delimiter}
(Britain{tuple_delimiter}the united kingdom{tuple_delimiter}true{tuple_delimiter}Britain and the United Kingdom are often used interchangeably to refer to the same sovereign state, although technically the United Kingdom includes Northern Ireland as well.{tuple_delimiter}0.95){record_delimiter}
(Britain{tuple_delimiter}united kingdom{tuple_delimiter}true{tuple_delimiter}Britain and United Kingdom are often used interchangeably to refer to the same sovereign state, although technically the United Kingdom includes Northern Ireland as well.{tuple_delimiter}0.95){completion_delimiter}



#############################
-Real data-
######################
context: {context}
pairs of entities: {entity_pairs}
ouput: 
"""

ENTITY_LINK_REVIEW_PROMPT = PromptTemplate(
    input_variables=["context", "entity_pairs", "tuple_delimiter", "record_delimiter", "completion_delimiter"],
    template=ENTITY_LINK_REVIEW_TEMPLATE
).partial(
    tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
    record_delimiter=DEFAULT_RECORD_DELIMITER,
    completion_delimiter=DEFAULT_COMPLETION_DELIMITER
)


IRCOT_TEMPLATE = """
Task: Based on the provided background knowledge, answer the question step by step. If the question ultimately remains unanswered, generate a new question combining the information already answered, and return the main subject(s) of the new question.

Requirements:
- The information provided is factual and authoritative; you are not to question or attempt to modify this information with your own knowledge.
- If the question has not been answered, you must include a `new question` label in your response.
- If the provided background knowledge contains no useful information, the new question should remain as consistent as possible with the original question, aside from the standardization of entity names.
- Make a judgment based on your understanding; if the provided background knowledge can answer the question, output answer directly as short as possible.
- Your response must follow the JSON code format below, where `answer` represents the response or the new question, and `subjects` lists the main subject(s) of the new question.

{{
"answer": str,
"subjects": List[str]
}}

Example:
{examples}


Question: {question}
Background: {background}
Answer:
"""

IRCOT_PROMPT_EN = PromptTemplate(
    input_variables=["examples", "question", "background"],
    template=IRCOT_TEMPLATE
)

IRCOT_EXAMPLE_PROMPT_EN = PromptTemplate.from_template(
    "Question: {question}\nBackground: {background}\nAnswer: {output}"
)