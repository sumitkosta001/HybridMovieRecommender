"""
Sample Dataset for Movie Recommendation Engine.
Contains curated metadata across distinct genres, directors, and plot keywords.
"""

SAMPLE_MOVIES = [
    {
        "movie_id": 101,
        "title": "Interstellar",
        "genres": "Sci-Fi|Adventure|Drama",
        "director": "Christopher Nolan",
        "cast": "Matthew McConaughey Anne Hathaway Jessica Chastain",
        "release_year": 2014,
        "vote_average": 8.7,
        "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival across deep space, black holes, and relativistic time dilation."
    },
    {
        "movie_id": 102,
        "title": "Inception",
        "genres": "Sci-Fi|Action|Thriller",
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio Joseph Gordon-Levitt Elliot Page",
        "release_year": 2010,
        "vote_average": 8.8,
        "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the subconscious mind of a CEO."
    },
    {
        "movie_id": 103,
        "title": "The Dark Knight",
        "genres": "Action|Crime|Drama",
        "director": "Christopher Nolan",
        "cast": "Christian Bale Heath Ledger Aaron Eckhart",
        "release_year": 2008,
        "vote_average": 9.0,
        "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
    },
    {
        "movie_id": 104,
        "title": "Arrival",
        "genres": "Sci-Fi|Drama|Mystery",
        "director": "Denis Villeneuve",
        "cast": "Amy Adams Jeremy Renner Forest Whitaker",
        "release_year": 2016,
        "vote_average": 7.9,
        "overview": "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world, exploring nonlinear time perception."
    },
    {
        "movie_id": 105,
        "title": "Blade Runner 2049",
        "genres": "Sci-Fi|Mystery|Drama",
        "director": "Denis Villeneuve",
        "cast": "Ryan Gosling Harrison Ford Ana de Armas",
        "release_year": 2017,
        "vote_average": 8.0,
        "overview": "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who's been missing for thirty years in a dystopian future."
    },
    {
        "movie_id": 106,
        "title": "Dune: Part One",
        "genres": "Sci-Fi|Adventure|Action",
        "director": "Denis Villeneuve",
        "cast": "Timothée Chalamet Rebecca Ferguson Oscar Isaac",
        "release_year": 2021,
        "vote_average": 8.0,
        "overview": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset on a hazardous desert planet inhabited by giant sandworms."
    },
    {
        "movie_id": 107,
        "title": "The Matrix",
        "genres": "Sci-Fi|Action",
        "director": "Lana Wachowski Lilly Wachowski",
        "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss",
        "release_year": 1999,
        "vote_average": 8.7,
        "overview": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."
    },
    {
        "movie_id": 108,
        "title": "Pulp Fiction",
        "genres": "Crime|Drama",
        "director": "Quentin Tarantino",
        "cast": "John Travolta Uma Thurman Samuel L. Jackson",
        "release_year": 1994,
        "vote_average": 8.9,
        "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."
    },
    {
        "movie_id": 109,
        "title": "Django Unchained",
        "genres": "Western|Drama|Action",
        "director": "Quentin Tarantino",
        "cast": "Jamie Foxx Christoph Waltz Leonardo DiCaprio",
        "release_year": 2012,
        "vote_average": 8.5,
        "overview": "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner."
    },
    {
        "movie_id": 110,
        "title": "Inglourious Basterds",
        "genres": "War|Drama|Action",
        "director": "Quentin Tarantino",
        "cast": "Brad Pitt Christoph Waltz Diane Kruger",
        "release_year": 2009,
        "vote_average": 8.4,
        "overview": "In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theater owner's vengeful plans."
    },
    {
        "movie_id": 111,
        "title": "Spirited Away",
        "genres": "Animation|Adventure|Fantasy",
        "director": "Hayao Miyazaki",
        "cast": "Rumi Hiiragi Miyu Irino Mari Natsuki",
        "release_year": 2001,
        "vote_average": 8.6,
        "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts."
    },
    {
        "movie_id": 112,
        "title": "Princess Mononoke",
        "genres": "Animation|Adventure|Fantasy",
        "director": "Hayao Miyazaki",
        "cast": "Yoji Matsuda Yuriko Ishida Yuko Tanaka",
        "release_year": 1997,
        "vote_average": 8.4,
        "overview": "On a journey to find the cure for a Tatarigami's curse, Ashitaka finds himself in the middle of a war between the forest gods and Tatara, a mining colony."
    },
    {
        "movie_id": 113,
        "title": "Spider-Man: Into the Spider-Verse",
        "genres": "Animation|Action|Adventure",
        "director": "Bob Persichetti Peter Ramsey",
        "cast": "Shameik Moore Jake Johnson Hailee Steinfeld",
        "release_year": 2018,
        "vote_average": 8.4,
        "overview": "Teenager Miles Morales becomes the new Spider-Man and joins other Spider-Heroes from parallel dimensions to stop a threat to all realities."
    },
    {
        "movie_id": 114,
        "title": "The Grand Budapest Hotel",
        "genres": "Comedy|Drama|Adventure",
        "director": "Wes Anderson",
        "cast": "Ralph Fiennes Tony Revolori Saoirse Ronan",
        "release_year": 2014,
        "vote_average": 8.1,
        "overview": "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel's glorious years under an exceptional concierge."
    },
    {
        "movie_id": 115,
        "title": "Moonrise Kingdom",
        "genres": "Comedy|Drama|Romance",
        "director": "Wes Anderson",
        "cast": "Jared Gilman Kara Hayward Bruce Willis",
        "release_year": 2012,
        "vote_average": 7.8,
        "overview": "A pair of young lovers flee their New England island town, prompting a local search party to fan out and find them before a violent storm strikes."
    },
    {
        "movie_id": 116,
        "title": "Parasite",
        "genres": "Comedy|Drama|Thriller",
        "director": "Bong Joon Ho",
        "cast": "Song Kang-ho Lee Sun-kyun Cho Yeo-jeong",
        "release_year": 2019,
        "vote_average": 8.5,
        "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."
    },
    {
        "movie_id": 117,
        "title": "Whiplash",
        "genres": "Drama|Music",
        "director": "Damien Chazelle",
        "cast": "Miles Teller J.K. Simmons Paul Reiser",
        "release_year": 2014,
        "vote_average": 8.5,
        "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."
    },
    {
        "movie_id": 118,
        "title": "La La Land",
        "genres": "Comedy|Drama|Music|Romance",
        "director": "Damien Chazelle",
        "cast": "Ryan Gosling Emma Stone John Legend",
        "release_year": 2016,
        "vote_average": 8.0,
        "overview": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."
    },
    {
        "movie_id": 119,
        "title": "The Godfather",
        "genres": "Crime|Drama",
        "director": "Francis Ford Coppola",
        "cast": "Marlon Brando Al Pacino James Caan",
        "release_year": 1972,
        "vote_average": 9.2,
        "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."
    },
    {
        "movie_id": 120,
        "title": "Goodfellas",
        "genres": "Biography|Crime|Drama",
        "director": "Martin Scorsese",
        "cast": "Robert De Niro Ray Liotta Joe Pesci",
        "release_year": 1990,
        "vote_average": 8.7,
        "overview": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito."
    }
]

# Realistic user rating profiles
SAMPLE_RATINGS = [
    # User 1: Sci-Fi & Mind-Benders aficionado (Nolan & Villeneuve fan)
    {"user_id": 1, "movie_id": 101, "rating": 5.0},
    {"user_id": 1, "movie_id": 102, "rating": 5.0},
    {"user_id": 1, "movie_id": 104, "rating": 4.5},
    {"user_id": 1, "movie_id": 105, "rating": 4.5},
    {"user_id": 1, "movie_id": 107, "rating": 4.5},
    {"user_id": 1, "movie_id": 114, "rating": 2.5},

    # User 2: Crime, Gangster & Quentin Tarantino devotee
    {"user_id": 2, "movie_id": 108, "rating": 5.0},
    {"user_id": 2, "movie_id": 109, "rating": 4.5},
    {"user_id": 2, "movie_id": 110, "rating": 4.5},
    {"user_id": 2, "movie_id": 119, "rating": 5.0},
    {"user_id": 2, "movie_id": 120, "rating": 5.0},
    {"user_id": 2, "movie_id": 101, "rating": 3.0},

    # User 3: Animation & Whimsical Art enthusiast
    {"user_id": 3, "movie_id": 111, "rating": 5.0},
    {"user_id": 3, "movie_id": 112, "rating": 5.0},
    {"user_id": 3, "movie_id": 113, "rating": 4.5},
    {"user_id": 3, "movie_id": 114, "rating": 4.5},
    {"user_id": 3, "movie_id": 115, "rating": 4.0},
    {"user_id": 3, "movie_id": 108, "rating": 2.0},

    # User 4: Dramatic cinema & musical artistry
    {"user_id": 4, "movie_id": 117, "rating": 5.0},
    {"user_id": 4, "movie_id": 118, "rating": 5.0},
    {"user_id": 4, "movie_id": 116, "rating": 4.5},
    {"user_id": 4, "movie_id": 101, "rating": 4.0},
    {"user_id": 4, "movie_id": 103, "rating": 4.0},

    # User 5: Balanced modern cinema viewer
    {"user_id": 5, "movie_id": 102, "rating": 4.5},
    {"user_id": 5, "movie_id": 103, "rating": 5.0},
    {"user_id": 5, "movie_id": 106, "rating": 4.0},
    {"user_id": 5, "movie_id": 116, "rating": 5.0},
    {"user_id": 5, "movie_id": 117, "rating": 4.5},
    {"user_id": 5, "movie_id": 111, "rating": 4.0},

    # User 6: Action & Sci-Fi lover
    {"user_id": 6, "movie_id": 102, "rating": 4.5},
    {"user_id": 6, "movie_id": 103, "rating": 4.5},
    {"user_id": 6, "movie_id": 106, "rating": 4.5},
    {"user_id": 6, "movie_id": 107, "rating": 5.0},
    {"user_id": 6, "movie_id": 109, "rating": 4.0},
]
