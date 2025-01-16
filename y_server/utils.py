import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func, desc
from y_server.modals import db, Post, Post_topics, User_interest, Follow_status, Reactions, User_mgmt



def get_follows(uid):
    """
    Get the followers of a user.

    :param uid: the user id
    :return: a list of followers
    """
    # Get the latest round for each follower-user relationship
    res = Follow_status.query.filter_by(user_id=uid).all()
    res = [x.follower_id for x in res]

    return res

def print_posts(posts):
    """
    Print the post ids.

    :param posts: the posts query result
    """
    # print post id
    res = []
    for post_type in posts:
        for post in post_type:
            try:
                res.append(post[0].id)
            except:
                res.append(post.id)
    print(res, flush=True)

    return

def fetch_common_interest_posts(uid, visibility, articles, follower_posts_limit, additional_posts_limit):
    """
    Fetch posts based on common interests with followers and others.

    :param uid: the user id
    :param visibility: the visibility threshold
    :param articles: whether to include articles
    :param follower_posts_limit: the number of posts from followers
    :param additional_posts_limit: the number of additional posts
    :return: the posts query result
    """
    user_interests = db.session.query(User_interest.interest_id).filter_by(user_id=uid).distinct()
    follower_ids = get_follows(uid)

    base_query = (
        db.session.query(Post, func.count(Post_topics.topic_id).label('match_count'))
        .join(Post_topics, Post.id == Post_topics.post_id)
        .filter(
            Post.round >= visibility,
            Post.user_id != uid,
            Post.news_id != -1 if articles else True
        )
        .group_by(Post.id)
        .order_by(desc('match_count'))
    )

    # Helper to create queries
    def create_query(user_filter, topic_filter, lit):
        return base_query.filter(user_filter, topic_filter).limit(lit)

    posts = []

    # Posts by followers with common interests
    posts.append(create_query(
        Post.user_id.in_(follower_ids),
        Post_topics.topic_id.in_(user_interests),
        follower_posts_limit
    ).all())

    # Additional posts with common interests
    if additional_posts_limit > 0:
        posts.append(create_query(
            Post.user_id.notin_(follower_ids),
            Post_topics.topic_id.in_(user_interests),
            additional_posts_limit
        ).all())

    return posts

def fetch_common_user_interest_posts(uid, visibility, articles, limit, reactions_type:str|list=["like", "dislike"]):
    """
    Fetch posts reacted by users with common interests.

    :param uid: the user id
    :param visibility: the visibility threshold
    :param articles: whether to include articles
    :param limit: the number of posts to fetch
    :return: the posts query result
    """

    if isinstance(reactions_type, str):
        reactions_type = [reactions_type]
    
    # get users with common topic interests
    common_users_query = (
        db.session.query(User_mgmt.id, func.count(User_interest.interest_id).label("match_count"))
        .join(User_interest, User_mgmt.id == User_interest.user_id)
        .filter(
            User_mgmt.id != uid,
            User_interest.interest_id.in_(
                db.session.query(User_interest.interest_id).filter_by(user_id=uid)
            )
        )
        .group_by(User_mgmt.id)
        .order_by(desc("match_count"))
        .limit(limit)
        .subquery()
    )
    # fetch posts liked by users with common interests
    posts = [(
            db.session.query(Post, func.count(Reactions.user_id).label("total"))
            .join(Reactions, Post.id == Reactions.post_id) 
            .filter(
                Reactions.user_id.in_(
                    db.session.query(common_users_query.c.id)
                ),
                Reactions.type.in_(reactions_type),
                Post.round >= visibility,
                Post.news_id != -1 if articles else True
        )
        .group_by(Post)
        .order_by(desc("total"), desc(Post.id))
        .limit(limit)
    )]

    return posts

def fetch_knn_posts(uid, visibility, limit):
    """
    Fetch posts based on common interests with followers and others.

    :param uid: the user id
    :param visibility: the visibility threshold
    :param limit: the number of posts to fetch
    :return: the posts query result
    """

    # Step 1: Fetch all posts visible to the user
    posts = Post.query.filter(Post.round >= visibility).all()
    post_ids = [post.id for post in posts]
    post_texts = [post.tweet for post in posts]
    post_authors = [post.user_id for post in posts]

    # Step 2: Prepare TF-IDF features for post text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    text_features = vectorizer.fit_transform(post_texts).toarray()

    # Step 3: Add author metadata as features
    author_features = []
    for user_id in post_authors:
        user = User_mgmt.query.get(user_id)
        author_features.append(
            vectorizer.fit_transform([
                user.leaning +
                user.oe +
                user.co +
                user.ex +
                user.ag +
                user.ne +
                user.toxicity]).toarray() + [user.age]
        )
        #padding to make all author features of same length
        author_features[-1] = np.pad(author_features[-1][0], (0, 50 - author_features[-1].shape[1]), 'constant')
    author_features = np.array(author_features)

    # Combine text and author features
    combined_features = np.hstack((text_features, author_features))

    # Step 4: Compute cosine similarity between posts
    post_similarities = cosine_similarity(combined_features)

    # Step 5: Find recent posts interacted with by the user
    recent_interactions = Reactions.query.filter_by(user_id=uid).order_by(desc(Reactions.round)).limit(5).all()
    recent_post_ids = [interaction.post_id for interaction in recent_interactions]

    # Step 6: Recommend posts similar to the user's recent interactions
    recommended_post_ids = set()
    for recent_post_id in recent_post_ids:
        if recent_post_id in post_ids:
            idx = post_ids.index(recent_post_id)
            similar_indices = np.argsort(-post_similarities[idx])  # Descending order
            for sim_idx in similar_indices[1:]:  # Skip self
                recommended_post_ids.add(post_ids[sim_idx])
                if len(recommended_post_ids) >= limit:
                    break
        if len(recommended_post_ids) >= limit:
            break

    res = list(recommended_post_ids)[:limit]
    print(res, flush=True)
    res = [db.session.query(Post).filter(Post.id.in_(res)).all()]
    print(res, flush=True)
    return res