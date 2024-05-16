import pandas as pd
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import joblib

WINDOW_SIZE = 256
DATA_PATH = Path(__file__).parent / "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
if not (DATA_PATH / "movie_avg_ratings_dict.pkl").exists():
    movie_ratings = defaultdict(list)
    movie_id = ""
    for i in range(1, 5):
        print(i)
        with open(DATA_PATH / f"combined_data_{i}.txt") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id = line[:-1]
                else:
                    _, rating, _ = line.split(",")
                    movie_ratings[int(movie_id)].append(int(rating))

    movie_avg_rating_dict = pd.Series(movie_ratings).apply(np.mean).to_dict()
    joblib.dump(movie_avg_rating_dict, DATA_PATH / "movie_avg_ratings_dict.pkl")
else:
    movie_avg_rating_dict = joblib.load(DATA_PATH / "movie_avg_ratings_dict.pkl")


if not (DATA_PATH / "movie_ratings_df.pkl").exists():
    data = defaultdict(list)
    movie_id = ""
    user_list = ["305344", "387418", "2439493", "1664010", "2118461"]

    for i in range(1, 5):
        print(i)
        with open(DATA_PATH / f"combined_data_{i}.txt") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id = line[:-1]
                else:
                    user_id, rating, date = line.split(",")
                    if user_id in user_list:

                        data["movie_id"].append(int(movie_id))
                        data["user_id"].append(int(user_id))
                        data["rating"].append(int(rating))
                        data["date"].append(date)

    ratings_df = pd.DataFrame(data)
    ratings_df["date"] = pd.to_datetime(ratings_df["date"])
    joblib.dump(ratings_df, DATA_PATH / "movie_ratings_df.pkl")
else:
    ratings_df = joblib.load(DATA_PATH / "movie_ratings_df.pkl")


# https://github.com/tommasocarraro/netflix-prize-with-genres/blob/master/netflix_genres.csv
genres_df = pd.read_csv(DATA_PATH / "netflix_genres.csv")

genres_df["genres"] = genres_df["genres"].str.split("|")
genres = genres_df["genres"].explode().value_counts().index
genres_df[genres] = 0
for i in genres:
    genres_df[i] = genres_df["genres"].apply(lambda x: i in x).astype(int)


movies_df = (
    ratings_df[["movie_id"]]
    .drop_duplicates()
    .merge(genres_df.rename(columns={"movieId": "movie_id"}))
    .sort_values("movie_id", ignore_index=True)
    .rename_axis("movie_rank", axis=0)
    .reset_index()
)
movie_id_rank_mapping = movies_df.set_index("movie_id")["movie_rank"].to_dict()
movies_df["avg_movie_rating"] = movies_df.apply(
    lambda x: movie_avg_rating_dict[x["movie_id"]] / 5, axis=1
)

MOVIE_FEATURES = torch.from_numpy(
    movies_df[genres.to_list() + ["avg_movie_rating"]]
    .values[np.newaxis, ...]
    .astype(np.float32)
).to(device)


class RatingDataset(Dataset):
    def __init__(
        self,
        rating_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
    ):
        self.df = rating_df.merge(
            movies_df.drop(columns=["genres"]), on="movie_id"
        ).sort_values(by=["user_id", "date", "movie_id"], ignore_index=True)
        self.user_counts = self.df["user_id"].value_counts().sort_index()
        self.window_size = window_size + 1
        self.state = np.stack(
            self.df.apply(
                lambda x: x[genres].to_list()
                + [x["avg_movie_rating"]]
                + [x["rating"] / 5],
                axis=1,
            ),
            dtype=np.float32,
        )
        self.action = self.df["movie_rank"].values.astype(int)
        self.reward = np.array(self.df["rating"] / 5, dtype=np.float32)

    def __len__(self):
        return len(self.df) - len(self.user_counts) * (self.window_size - 1)

    def __getitem__(self, idx):
        offset = 0
        for num_rows in self.user_counts:
            if idx > num_rows + offset - self.window_size:
                offset += num_rows
                idx += self.window_size - 1
            else:
                return (
                    self.state[idx : idx + self.window_size - 1],
                    self.action[idx + self.window_size - 1],
                    self.reward[idx + self.window_size - 1],
                    self.state[idx + 1 : idx + self.window_size],
                )
