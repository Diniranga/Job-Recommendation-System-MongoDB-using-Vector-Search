require('dotenv').config();
const fs = require("fs");
const pdf = require("pdf-parse");
const readline = require("readline");
const { MongoClient, ServerApiVersion} = require("mongodb");
const { HfInference } = require("@huggingface/inference");

const allJobPostings = require("./jobPostings");
const { MONGO_HOST, MONGO_USER, MONGO_PASS, MONGO_DB , MONGO_COLLECTION } = process.env;
const uri = `mongodb+srv://${MONGO_USER}:${MONGO_PASS}@${MONGO_HOST}/?retryWrites=true&w=majority`;

const hf = new HfInference("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");

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

async function extractTextFromPDF(filePath) {
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdf(dataBuffer);
        return data.text.replace(/\n/g, " ").replace(/ +/g, " ");
    } catch (err) {
        console.error(err.message);
    }
}

async function generateEmbeddings(text) {
    try {
        const result = await hf.featureExtraction({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            inputs: text,
        });
        // console.log("Embedding API result:", result);
        return result
    } catch (err) {
        console.error(err.message);
    }
}

const promptUserInput = (query) => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    return new Promise((resolve) =>
        rl.question(query, (answer) => {
            rl.close();
            resolve(answer);
        })
    );
};

async function storeEmbeddings (collection, jobPostings) {

    const jobEmbeddings = [];
    for (const job of jobPostings) {
        const embedding = await generateEmbeddings(job.jobDescription.toLowerCase());
        jobEmbeddings.push(embedding);
    }

    const jobsWithEmbeddings = jobPostings.map((job, index) => ({
        ...job,
        embedding: jobEmbeddings[index],
    }));
    await collection.insertMany(jobsWithEmbeddings);
}

async function performSimilaritySearch(collection, queryTerm) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);

        const pipeline = [
            {
                '$vectorSearch': {
                    'index': 'smart_job_recommend_data',
                    'path': 'embedding',
                    'queryVector': queryEmbedding[0],
                    'numCandidates': 50,
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
                '$sort': {
                    'score': -1
                }
            }
        ];

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
    try {
        await connectToMongoDB();
        const db = client.db(MONGO_DB);
        const collection = db.collection(MONGO_COLLECTION);

        await storeEmbeddings(collection, allJobPostings );

        const filePath = await promptUserInput("Enter file path to PDF: ");
        const text = await extractTextFromPDF("testResume.pdf");

        const initialResults = await performSimilaritySearch(collection, text );

        initialResults.forEach((item, index) => {
            console.log(`Top ${index + 1} Recommended Job Name: ${item.job_name}`);
        });


    } catch (err) {
        console.error(err.message);
    }
}

main();
