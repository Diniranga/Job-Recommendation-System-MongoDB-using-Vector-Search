require('dotenv').config();
const { MongoClient, ServerApiVersion} = require("mongodb");
const { HfInference } = require("@huggingface/inference");

const { MONGO_HOST, MONGO_USER, MONGO_PASS, MONGO_DB , MONGO_COLLECTION } = process.env;

const uri = `mongodb+srv://${MONGO_USER}:${MONGO_PASS}@${MONGO_HOST}/?retryWrites=true&w=majority`;
const hf = new HfInference("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");

const jobPosts = require("./jobPostings");

let client;

async function connectToMongoDB() {
    if (!client) {
        client = new MongoClient(uri, {
            serverApi: {
                version: ServerApiVersion.v1,
                strict: false,
                deprecationErrors: true,
            }
        });
        try {
            await client.connect();
            console.log('Connected to MongoDB Atlas');
        } catch (err) {
            console.error('Error connecting to MongoDB:', err);
            throw err;
        }
    }
    return client.db(MONGO_DB);
}

async function closeMongoDBConnection() {
    if (client) {
        await client.close();
        console.log('MongoDB connection closed');
        client = null;
    }
}

async function generateEmbeddings(text) {
    try {
        return await hf.featureExtraction({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            inputs: text,
        });
    } catch (err) {
        console.error('Error generating Embeddings:', err);
    }
}

async function storeEmbeddings (collection, jobPostings, embeddings) {
    const jobsWithEmbeddings = jobPostings.map((job, index) => ({
        ...job,
        embedding: embeddings[index],
    }));
    await collection.insertMany(jobsWithEmbeddings);
}

async function classifyText(text, labels){
    const response = await hf.request({
        model: "facebook/bart-large-mnli",
        inputs: text,
        parameters: {
            candidate_labels: labels
        }
    });
    return response;
}

async function extractFilterCriteria (query) {
    const criteria = { location: null, jobTitle: null, company: null, jobType: null };
    const labels = ["location", "job title", "company", "job type"];

    const words = query.split(" ");
    for(const word of words) {
        const result = await classifyText(word, labels);
        console.log(result);
        const highestScoreLabel = result.labels[0];
        const score = result.scores[0];
        if(score > 0.4) {
            switch (highestScoreLabel) {
                case "location":
                    criteria.location = word;
                    break;
                case "job title":
                    criteria.jobTitle = word;
                    break;
                case "company":
                    criteria.company = word;
                    break;
                case "job type":
                    criteria.jobType = word;
                    break;
                default:
                    break;
            }
        }

    }
    return criteria;
}


async function performSimilaritySearch(collection, queryTerm, filteredCriteria) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);

        const pipeline = [
            {
                '$vectorSearch': {
                    'index': 'job_vector_search',
                    'path': 'embedding',
                    'queryVector': queryEmbedding[0],
                    'numCandidates': 38,
                    'limit': 5
                }
            },
            {
                '$set': {
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            },
            {
                '$match': {
                    '$or': []
                }
            },
            {
                '$sort': {
                    'score': -1
                }
            }
        ];

        // // Add filter conditions based on filteredCriteria
        // if (filteredCriteria.location) {
        //     pipeline[2]['$match']['$or'].push({ 'location': { '$regex': filteredCriteria.location } });
        // }
        // if (filteredCriteria.jobTitle) {
        //     pipeline[2]['$match']['$or'].push({ 'jobTitle': { '$regex': filteredCriteria.jobTitle } });
        // }
        // if (filteredCriteria.company) {
        //     pipeline[2]['$match']['$or'].push({ 'company': { '$regex': filteredCriteria.company, } });
        // }
        // if (filteredCriteria.jobType) {
        //     pipeline[2]['$match']['$or'].push({ 'jobType': { '$regex': filteredCriteria.jobType, } });
        // }

        // If no criteria were added, remove the $match stage
        if (pipeline[2]['$match']['$or'].length === 0) {
            pipeline.splice(2, 1);
        }

        const results = await collection.aggregate(pipeline).toArray();

        if (!results || results.length === 0) {
            console.log(`No Job items found similar to "${queryTerm}" with given criteria`);
            return [];
        }

        let topJobPosts = results.map(result => {
            return {
                jobId: result.jobId,
                score: result.score,
                job_name: result.jobTitle,
                job_description: result.jobDescription,
                location: result.location,
                job_type: result.jobType,
                company: result.company
            };
        });
        return topJobPosts;

    } catch (error) {
        console.log(error);
    }
}

async function main() {

    const query = "Python";

    try {
        await connectToMongoDB();
        const db = client.db(MONGO_DB);
        const collection = db.collection(MONGO_COLLECTION);
        //
        // const jobTexts = jobPosts.map(jobPost => `${jobPost.jobTitle}, ${jobPost.jobDescription}, ${jobPost.jobType}, ${jobPost.location}`)
        // const jobDataEmbeddings = [];
        // for (let jobText of jobTexts) {
        //     const embedding = await generateEmbeddings(jobText);
        //     jobDataEmbeddings.push(embedding);
        // }
        // await storeEmbeddings(collection, jobPosts, jobDataEmbeddings);

        const filteredCriteria = await extractFilterCriteria(query);
        const initialResults = await performSimilaritySearch(collection, query, filteredCriteria );

        initialResults.forEach((item, index) => {
            console.log(`Top ${index + 1} Recommended Job Name: ${item.job_name}`);
        });

    }catch (error) {
        console.log(error);
    }
}

main();