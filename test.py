TEXT = """
Sundar Pichai is the CEO of Google. Tim Cook is the CEO of Apple. Satya Nadella is the CEO of Microsoft. Andy Jassy is the CEO of Amazon. Jensen Huang is the CEO of Nvidia.
Google is headquartered in Mountain View. Apple is headquartered in Cupertino. Microsoft is headquartered in Redmond. Amazon is headquartered in Seattle. Nvidia is headquartered in Santa Clara.
Mountain View, Cupertino, and Santa Clara are cities in California. Seattle and Redmond are cities in Washington. California and Washington are states in the United States. The United States is a country in North America. India is a country in Asia. The United Kingdom is a country in Europe. Japan is a country in Asia.
Google has major offices in London, New York City, Bengaluru, and Hyderabad. Microsoft has offices in London, Hyderabad, and Tokyo. Amazon has offices in Bengaluru, Hyderabad, London, and New York City. Apple has offices in Cupertino, London, and Tokyo. Nvidia has offices in Bengaluru, Hyderabad, Tokyo, and London.
London is the capital of the United Kingdom. New Delhi is the capital of India. Tokyo is the capital of Japan. Washington, D.C. is the capital of the United States.
The River Thames flows through London. The Hudson River flows through New York City. The Yamuna flows through New Delhi. The Musi River flows through Hyderabad.
Big Ben is located in London. Tower Bridge is located in London. The Statue of Liberty is located in New York City. India Gate is located in New Delhi. Charminar is located in Hyderabad. Tokyo Tower is located in Tokyo.
Bengaluru is in Karnataka, Hyderabad is in Telangana, Mumbai is in Maharashtra, New Delhi is in Delhi, and Tokyo is in Tokyo Prefecture. Karnataka, Telangana, Maharashtra, and Delhi are in India. Tokyo Prefecture is in Japan.
India celebrates Diwali and Holi. The United States celebrates Thanksgiving. The United Kingdom celebrates Christmas. Japan celebrates Golden Week.
Infosys is headquartered in Bengaluru. TCS is headquartered in Mumbai. Wipro is headquartered in Bengaluru. OpenAI is headquartered in San Francisco. San Francisco is a city in California. Sam Altman leads OpenAI.
Infosys has offices in Hyderabad, Bengaluru, London, and New York City. TCS has offices in Mumbai, Hyderabad, London, and Tokyo. Wipro has offices in Bengaluru, Hyderabad, London, and New York City. OpenAI has an office in San Francisco.
Google operates in the United States, India, the United Kingdom, and Japan. Microsoft operates in the United States, India, the United Kingdom, and Japan. Amazon operates in the United States, India, the United Kingdom, and Japan. Apple operates in the United States, the United Kingdom, India, and Japan. Nvidia operates in the United States, India, the United Kingdom, and Japan.
California borders Oregon, Nevada, and Arizona. Washington borders Oregon and Idaho. Telangana is traversed by the Godavari and Krishna rivers. Karnataka is known for Bengaluru. Telangana is known for Hyderabad. Maharashtra is known for Mumbai.
The Gateway of India is located in Mumbai. Marine Drive is located in Mumbai. The Space Needle is located in Seattle. The Golden Gate Bridge is located in San Francisco.
Mumbai is a major city in India. New York City is a major city in the United States. London is a major city in the United Kingdom. Tokyo is a major city in Japan. Bengaluru and Hyderabad are major technology hubs in India.
""".strip()

ENTITIES = [
    "Person",
    "Organization",
    "City",
    "State",
    "Country",
    "Continent",
    "River",
    "Monument",
    "Festival",
]

RELATIONS = [
    "LEADS",
    "HEADQUARTERED_IN",
    "HAS_OFFICE",
    "OPERATES_IN",
    "LOCATED_IN",
    "CAPITAL_OF",
    "HAS_CAPITAL",
    "FLOWS_THROUGH",
    "HAS_MONUMENT",
    "CELEBRATES",
    "BORDERS",
    "KNOWN_FOR",
]

POTENTIAL_SCHEMA = [
    ("Person", "LEADS", "Organization"),
    ("Organization", "HEADQUARTERED_IN", "City"),
    ("Organization", "HAS_OFFICE", "City"),
    ("Organization", "OPERATES_IN", "Country"),
    ("City", "LOCATED_IN", "State"),
    ("State", "LOCATED_IN", "Country"),
    ("Country", "LOCATED_IN", "Continent"),
    ("City", "CAPITAL_OF", "Country"),
    ("Country", "HAS_CAPITAL", "City"),
    ("River", "FLOWS_THROUGH", "City"),
    ("River", "FLOWS_THROUGH", "State"),
    ("Monument", "LOCATED_IN", "City"),
    ("City", "HAS_MONUMENT", "Monument"),
    ("Country", "CELEBRATES", "Festival"),
    ("State", "KNOWN_FOR", "City"),
    ("Country", "KNOWN_FOR", "City"),
    ("State", "BORDERS", "State"),
]