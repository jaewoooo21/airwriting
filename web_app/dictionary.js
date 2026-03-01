/**
 * dictionary.js — Lightweight Autocorrect Engine for AirWriting
 * Uses Levenshtein edit distance to find the closest known word.
 */

const DICTIONARY = [
    // Common English words (sorted by frequency)
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I",
    "IT", "FOR", "NOT", "ON", "WITH", "HE", "AS", "YOU", "DO", "AT",
    "THIS", "BUT", "HIS", "BY", "FROM", "THEY", "WE", "SAY", "HER", "SHE",
    "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD", "THERE", "THEIR", "WHAT",
    "SO", "UP", "OUT", "IF", "ABOUT", "WHO", "GET", "WHICH", "GO", "ME",
    "WHEN", "MAKE", "CAN", "LIKE", "TIME", "NO", "JUST", "HIM", "KNOW", "TAKE",
    "PEOPLE", "INTO", "YEAR", "YOUR", "GOOD", "SOME", "COULD", "THEM", "SEE",
    "OTHER", "THAN", "THEN", "NOW", "LOOK", "ONLY", "COME", "ITS", "OVER",
    "THINK", "ALSO", "BACK", "AFTER", "USE", "TWO", "HOW", "OUR", "WORK",
    "FIRST", "WELL", "WAY", "EVEN", "NEW", "WANT", "BECAUSE", "ANY", "THESE",
    "GIVE", "DAY", "MOST", "US", "GREAT", "BETWEEN", "NEED", "LARGE",
    // Common test/demo words
    "HELLO", "WORLD", "APPLE", "STAR", "MOON", "SUN", "FIRE", "WATER",
    "AIR", "WRITE", "WRITING", "TEST", "CODE", "LOVE", "LIFE", "GAME",
    "PLAY", "HOME", "HAND", "OPEN", "CLOSE", "START", "STOP", "DONE",
    "HELP", "NAME", "CALL", "BOOK", "READ", "TELL", "KEEP", "HEAD",
    "MOVE", "TURN", "LEFT", "RIGHT", "HIGH", "LONG", "SMALL", "BIG",
    "OLD", "YOUNG", "LAST", "NEXT", "SAME", "BOTH", "FEW", "MUCH",
    "CAT", "DOG", "BIRD", "FISH", "TREE", "RAIN", "SNOW", "WIND",
    "KING", "QUEEN", "NIGHT", "LIGHT", "DARK", "DREAM", "HOPE", "BEST",
    // Alphabet letters (single)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
];

/**
 * Compute Levenshtein edit distance between two strings.
 */
function levenshtein(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            dp[i][j] = Math.min(
                dp[i - 1][j] + 1,       // delete
                dp[i][j - 1] + 1,       // insert
                dp[i - 1][j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0) // replace
            );
        }
    }
    return dp[m][n];
}

/**
 * Find the closest word(s) from the dictionary.
 * Returns an array of { word, distance } sorted by distance.
 */
function findClosestWords(input, maxResults = 3) {
    if (!input || input.length === 0) return [];
    const upper = input.toUpperCase();

    // Exact match → no correction needed
    if (DICTIONARY.includes(upper)) {
        return [{ word: upper, distance: 0 }];
    }

    let results = DICTIONARY
        .map(w => ({ word: w, distance: levenshtein(upper, w) }))
        .filter(r => r.distance <= Math.max(2, Math.floor(upper.length / 2))) // Max 50% of length
        .sort((a, b) => a.distance - b.distance)
        .slice(0, maxResults);

    // Deduplicate
    const seen = new Set();
    return results.filter(r => {
        if (seen.has(r.word)) return false;
        seen.add(r.word);
        return true;
    });
}
