from py2neo import Graph, Node, Relationship, authenticate
import credentials

authenticate("localhost:7474", credentials.user, credentials.password)
graph = Graph('http://localhost:7474/db/data/')
query = ("MATCH (m1:Movie) WITH count(m1) as countm RETURN countm")

tx = graph.cypher.begin()

def add_new_users():
    print("Not yet")

query_no_users = ( 'MATCH (u1:`User`)'
        'WITH count(u1) as count_users '
        'RETURN count_users')

query_no_movies = ('MATCH (m1:`Movie`)'
        'WITH count(m1) as count_movies '
        'RETURN count_movies')

query_no_ratings_by_user = ('MATCH (u1:`User`)-[:`RATED`]->(m1:`Movie`) '
                            'WITH u1,count(m1) AS number_rated_movies  '
                            'RETURN sum(number_rated_movies) as total_ratings')

def get_data_query(query):
    tx = graph.cypher.begin()
    tx.append(query)
    result = tx.commit()
    return result


def reccomend_by_similar_ratings(user_id, threshold, rec_number = 10):
    query = (  ### Similarity normalization : count number of movies seen by u1 ###
        # Count movies rated by u1 as countm
        'MATCH (m1:`Movie`)<-[:`RATED`]-(u1:`User` {id:{user_id}}) '
        'WITH count(m1) as countm '
        ### Recommendation ###
        # Retrieve all users u2 who share at least one movie with u1
        'MATCH (u2:`User`)-[r2:`RATED`]->(m1:`Movie`)<-[r1:`RATED`]-(u1:`User` {id:{user_id}}) '
        # Check if the ratings given by u1 and u2 differ by less than 1
        'WHERE (NOT u2=u1) AND (abs(r2.rating - r1.rating) <= 1) '
        # Compute similarity
        'WITH u1, u2, tofloat(count(DISTINCT m1))/countm as sim '
        # Keep users u2 whose similarity with u1 is above some threshold
        'WHERE sim>{threshold} '
        # Retrieve movies m that were rated by at least one similar user, but not by u1
        'MATCH (m:`Movie`)<-[r:`RATED`]-(u2) '
        'WHERE (NOT (m)<-[:`RATED`]-(u1)) '
        # Compute score and return the list of suggestions ordered by score
        'WITH DISTINCT m, count(r) as n_u, tofloat(sum(r.rating)) as sum_r '
        'WHERE n_u > 1 '
        'RETURN m, sum_r/n_u as score ORDER BY score DESC LIMIT {rec_number}')
    tx = graph.cypher.begin()
    tx.append(query, {'user_id': user_id, 'threshold': threshold, 'rec_number': rec_number})
    result = tx.commit()
    return result

result = reccomend_by_similar_ratings(139, 0.5)
print(result)

no_users = get_data_query(query_no_users)
print(no_users)

no_movies = get_data_query(query_no_movies)
print(no_movies)

no_queries_by_user = get_data_query(query_no_ratings_by_user)
print(no_queries_by_user)


"""
CYPHER SYNTAX
MATCH (m1:Movie)<-[:RATED]-(u1:User {id:139})
WITH count(m1) as countm 
MATCH (u2:User)-[r2:RATED]->(m1:Movie)<-[r1:RATED]-(u1:User {id:139})
WHERE (NOT u2=u1) AND (abs(r2.rating - r1.rating) <= 1)
WITH u1, u2, tofloat(count(DISTINCT m1))/countm as sim 
WHERE sim>0.5 
MATCH (m:Movie)<-[r:RATED]-(u2) 
WHERE (NOT (m)<-[:RATED]-(u1))
WITH DISTINCT m, count(r) as n_u, tofloat(sum(r.rating)) as sum_r 
WHERE n_u > 1 
RETURN m, sum_r/n_u as score ORDER BY score DESC
"""