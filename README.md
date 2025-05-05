# GiftFindr

Giftfindr is a prototype that turns natural-language gift queries into personalized recommendations using a vector search (FAISS) over a small product catalog.

Get helpful links and gift recommendations to users trying to find a gift. Given preferences, such as age, likes, and budget, the chatbot can give useful gift suggestions, so the user can directly go to the site to purchase. The user will have a convenient way to search for gifts by filtering on their needs.

To run:
    in root:
        npm install
        npm run build
        npm run dev
        open localhost
    in backend:
        install dependencies:
            pip install -r requirements.txt
        python server.py

        *We have already run FAISS, but there is a README in the backend folder if you'd like to run it yourself!