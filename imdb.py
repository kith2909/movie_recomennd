import requests


def get_api_data(url):
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error {response.status_code} occurred.")


def get_imdb(movies):
    api_base = "https://imdb-api.com/en/API/SearchMovie/k_u642fo4o/"
    info = []
    for movie_name in movies:

        api_url = api_base + movie_name  # Replace with the actual API endpoint URL
        api_data = get_api_data(api_url)
        print(api_data)
        # Process the API data
        if api_data:
            # Extract required information from the data
            info.append(api_data['results'])
            # Perform further operations with the extracted information
    return info


print(get_imdb(["Titanic"]))
