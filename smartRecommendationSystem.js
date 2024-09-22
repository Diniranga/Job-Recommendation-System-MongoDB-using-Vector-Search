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

const extractCVKeywords = async (text) => {
    // Convert text to lowercase for case-insensitive matching
    const lowercaseText = text.toLowerCase();

    // Define patterns for different sections of a CV
    const patterns = {
        skills: /skills?:?([\s\S]+?)(education|experience|projects?|$)/i,
        education: /education:?([\s\S]+?)(experience|skills?|projects?|$)/i,
        experience: /experience:?([\s\S]+?)(education|skills?|projects?|$)/i,
        projects: /projects?:?([\s\S]+?)(education|experience|skills?|$)/i
    };

    // Function to extract and clean keywords from a section
    const extractFromSection = (sectionText) => {
        return sectionText
            .split(/[,\s]+/) // Split on commas and whitespace
            .map(item => item.trim())
            .filter(item => item.length > 1 && !stopWords.has(item)); // Remove single-character items and stop words
    };

    // Common words to exclude
    const stopWords = new Set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with',
        'i', 'am', 'years', 'year', 'experience', 'using', 'used', 'developed', 'created'
    ]);

    // Extract keywords from each section
    let keywords = [];
    for (const [section, pattern] of Object.entries(patterns)) {
        const match = pattern.exec(lowercaseText);
        if (match) {
            keywords = keywords.concat(extractFromSection(match[1]));
        }
    }

    // Additional processing to extract more relevant information
    const jobTitles = extractJobTitles(lowercaseText);
    const technologies = extractTechnologies(lowercaseText);
    const years = extractYearsOfExperience(lowercaseText);

    return {
        jobTitles: jobTitles,
        technologies: technologies,
        yearsOfExperience: years
    };
};

// Helper function to extract job titles
const extractJobTitles = (text) => {
    const titlePattern = /(frontend|backend|full stack|software|web) (developer|engineer)/g;
    return [...new Set(text.match(titlePattern) || [])];
};

// Helper function to extract technologies
const extractTechnologies = (text) => {
    const techPattern = /(react|node|express|javascript|html|css|mongodb|firebase|git|github|tailwind|redux|next\.js)/g;
    return [...new Set(text.match(techPattern) || [])];
};

// Helper function to extract years of experience
const extractYearsOfExperience = (text) => {
    const yearsPattern = /(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*experience/i;
    const match = text.match(yearsPattern);
    return match ? parseFloat(match[1]) : null;
};

async function createCVString(cvInfo) {
    const parts = [];

    // Add job titles
    if (cvInfo.jobTitles && cvInfo.jobTitles.length > 0) {
        parts.push(`Job Titles: ${cvInfo.jobTitles.join(', ')}`);
    }

    // Add technologies
    if (cvInfo.technologies && cvInfo.technologies.length > 0) {
        parts.push(`Technologies: ${cvInfo.technologies.join(', ')}`);
    }

    // Add years of experience
    if (cvInfo.yearsOfExperience) {
        parts.push(`Years of Experience: ${cvInfo.yearsOfExperience}`);
    }

    // Combine all parts
    return parts.join('. ');
}

async function performSimilaritySearch(collection, queryTerm) {
    try {
        // Extract keywords and information from the CV
        const cvInfo = extractCVKeywords(queryTerm);
        console.log(cvInfo)

        // Generate embeddings for the CV keywords
        const cvInfoString = await createCVString(cvInfo);
        const queryEmbedding = await generateEmbeddings(cvInfoString);

        if (!queryEmbedding || queryEmbedding.length === 0) {
            console.error("Failed to generate query embedding");
            return [];
        }

        // Perform vector similarity search
        const pipeline = [
            {
                '$vectorSearch': {
                    'index': 'smart_job_recommend_data',
                    'path': 'embedding',
                    'queryVector': queryEmbedding,
                    'numCandidates': 100,
                    'limit': 20
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
            return [];
        }

        // Sort by total score (combination of vector similarity and keyword matching)
        results.sort((a, b) => b.score - a.score);

        // Take top 10 results
        return results.slice(0, 10).map(result => {
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

    } catch (error) {
        console.error("Error in performSimilaritySearch:", error);
        return [];
    }
}


async function main() {
    try {
        const db = await connectToMongoDB();
        const collection = db.collection(MONGO_COLLECTION);

        // Check if job postings exist in the database
        const jobCount = await collection.countDocuments();
        if (jobCount === 0) {
            console.log("No job postings found in the database. Inserting sample data...");
            await storeEmbeddings(collection, allJobPostings);
        }

        const filePath = await promptUserInput("Enter file path to PDF: ");
        const text = await extractTextFromPDF("testResume.pdf");
        if (!text) {
            throw new Error("Failed to extract text from PDF");
        }
        const initialResults = await performSimilaritySearch(collection, text);

        if (initialResults.length === 0) {
            console.log("No job recommendations found.");
        } else {
            console.log("Top Job Recommendations:");
            initialResults.forEach((item, index) => {
                console.log(`${index + 1}. ${item.job_name} (Score: ${item.score.toFixed(2)})`);
            });
        }

    } catch (err) {
        console.error("Error in main function:", err);
    } finally {
        await closeMongoDBConnection();
    }
}

main();