from flask import Flask, render_template, request, jsonify, make_response
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from urllib.parse import quote
from xhtml2pdf import pisa
from io import BytesIO
import re
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure executor for concurrent operations
executor = ThreadPoolExecutor(max_workers=10)

# Configure Google Gemini API
api_key = os.environ.get("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

RESPONSIVE_VOICE_KEY = os.environ.get("RESPONSIVE_VOICE_KEY")

# --- Helper Functions ---
def generate_image(prompt, width=800, height=600):
    try:
        encoded_prompt = quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
        return image_url
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return "https://image.pollinations.ai/prompt/scenic%20view?width=800&height=600&nologo=true"

def generate_all_images_concurrent(story_data):
    try:
        cover_prompt = f"A professional book cover with a title on it, '{story_data['title']}', fantasy art"
        cover_future = executor.submit(generate_image, cover_prompt, 400, 550)
        chapter_futures = [executor.submit(generate_image, chapter['image_prompt']) for chapter in story_data['chapters']]
        
        cover_image = cover_future.result()
        chapter_images = [future.result() for future in chapter_futures]
        
        for chapter, image_url in zip(story_data['chapters'], chapter_images):
            chapter['image'] = image_url
        
        return story_data, cover_image
    except Exception as e:
        print(f"Error in generate_all_images_concurrent: {str(e)}")
        fallback_url = "https://image.pollinations.ai/prompt/scenic%20view?width=800&height=600&nologo=true"
        cover_fallback = "https://image.pollinations.ai/prompt/book%20cover?width=400&height=550&nologo=true"
        
        for chapter in story_data['chapters']:
            chapter['image'] = fallback_url
        
        return story_data, cover_fallback

def get_word_definitions(words):
    prompt = f"""
    Define the following words clearly and concisely, focusing on their most common meanings in everyday usage.
    Words to define: {', '.join(words)}

    Respond ONLY with a JSON object in this exact format (no other text):
    {{
        "word1": "simple definition here",
        "word2": "simple definition here"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if '```json' in response_text:
            response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
        elif '```' in response_text:
            response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
        
        response_text = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)
        definitions = json.loads(response_text)
        
        final_definitions = {}
        for word in words:
            word_lower = word.lower()
            matching_key = next((k for k in definitions.keys() if k.lower() == word_lower), None)
            if matching_key and definitions[matching_key].strip():
                final_definitions[word] = definitions[matching_key]
            else:
                single_word_prompt = f"Define this word clearly and concisely in 10-15 words: {word}"
                try:
                    retry_response = model.generate_content(single_word_prompt)
                    definition = retry_response.text.strip()
                    final_definitions[word] = definition if definition else "No definition available"
                except Exception:
                    final_definitions[word] = "No definition available"
                    
        return final_definitions
        
    except Exception as e:
        print(f"Error getting definitions: {str(e)}")
        return {word: "Definition not available" for word in words}

def extract_terminology(text):
    words = re.findall(r'\b[A-Za-z]{5,}\b', text)
    
    common_words = {
        'there', 'their', 'would', 'could', 'should', 'about', 'which', 'these', 
        'those', 'were', 'have', 'that', 'what', 'when', 'where', 'while', 'from',
        'been', 'being', 'other', 'another', 'every', 'everything', 'something',
        'anything', 'nothing', 'through', 'although', 'though', 'without', 'within'
    }
    
    filtered_words = [
        word for word in words
        if word.lower() not in common_words
        and not word.isupper()
        and len(word) >= 6
        and not any(char.isdigit() for char in word)
    ]
    
    unique_words = []
    seen_words = set()
    for word in filtered_words:
        if word.lower() not in seen_words:
            unique_words.append(word)
            seen_words.add(word.lower())
    
    selected_words = sorted(unique_words, key=lambda x: (len(x), x.lower()), reverse=True)[:5]
    
    if selected_words:
        definitions = get_word_definitions(selected_words)
        return {k: v for k, v in definitions.items() if v and v != "No definition available"}
    return {}

# --- Story Generation and Processing Functions ---
def generate_story(keywords, genre, num_chapters, tone, style, age_group, include_magic, include_romance, include_conflict):
    prompt = f"""
    Create a captivating story based on the following:
    Keywords: {keywords}
    Genre: {genre}
    Number of chapters: {num_chapters}
    Tone: {tone}
    Writing Style: {style}
    Target Audience Age Group: {age_group}
    Include Magical Elements: {'Yes' if include_magic else 'No'}
    Include Romance: {'Yes' if include_romance else 'No'}
    Include Major Conflicts: {'Yes' if include_conflict else 'No'}

    Format the response as JSON with title, author, moral, and chapters array.
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        try:
            story_data = json.loads(response_text)
        except json.JSONDecodeError:
            if '```json' in response_text:
                response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
            elif '```' in response_text:
                response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
            
            try:
                story_data = json.loads(response_text)
            except:
                story_data = {
                    "title": f"{genre.title()} Story",
                    "author": "AI Author",
                    "moral": "Story generation failed",
                    "chapters": [{
                        "chapter_number": 1,
                        "chapter_title": "Chapter 1",
                        "content": "Failed to generate story. Please try again.",
                        "image_prompt": "A blank page",
                        "terminology": {}
                    }]
                }

        for chapter in story_data['chapters']:
            chapter['terminology'] = extract_terminology(chapter['content'])

        return story_data

    except Exception as e:
        print(f"Error generating story: {str(e)}")
        return {
            "title": f"{genre.title()} Story",
            "author": "AI Author",
            "moral": "Story generation failed",
            "chapters": [{
                "chapter_number": 1,
                "chapter_title": "Chapter 1",
                "content": "Failed to generate story. Please try again.",
                "image_prompt": "A blank page",
                "terminology": {}
            }]
        }

# --- API Routes ---
@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        story_length_map = {
            'short': 3,
            'medium': 5,
            'long': 7,
            'grand': 10
        }
        
        num_chapters = story_length_map.get(data.get('storyLength'), 3)
        
        story = generate_story(
            data.get('keywords'),
            data.get('genre'),
            num_chapters,
            data.get('tone'),
            data.get('style'),
            data.get('ageGroup'),
            data.get('includeMagic'),
            data.get('includeRomance'),
            data.get('includeConflict')
        )
        
        story, cover_image = generate_all_images_concurrent(story)
        
        return jsonify({
            'story': story,
            'cover_image': cover_image
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to generate story',
            'details': str(e)
        }), 500

@app.route('/api/regenerate', methods=['POST'])
def regenerate():
    try:
        data = request.get_json()
        story_data = data.get('story')
        chapter_number = int(data.get('chapter_number'))
        
        if not (0 <= chapter_number - 1 < len(story_data["chapters"])):
            return jsonify({'error': 'Invalid chapter number'}), 400
            
        prompt = f"""
        Regenerate chapter {chapter_number} of '{story_data['title']}'
        with the following parameters:
        Tone: {data.get('tone')}
        Style: {data.get('style')}
        Include Magic: {data.get('includeMagic')}
        Include Romance: {data.get('includeRomance')}
        Include Conflict: {data.get('includeConflict')}
        """
        
        response = model.generate_content(prompt)
        new_chapter = json.loads(response.text)
        
        new_chapter['terminology'] = extract_terminology(new_chapter['content'])
        story_data['chapters'][chapter_number - 1] = new_chapter
        new_chapter['image'] = generate_image(new_chapter['image_prompt'])
        
        return jsonify({'story': story_data, 'new_chapter': new_chapter})
    except Exception as e:
        return jsonify({
            'error': 'Failed to regenerate chapter',
            'details': str(e)
        }), 500

@app.route('/api/download', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        story_data = data.get('story')
        cover_image = data.get('cover_image')
        
        pdf_buffer = BytesIO()
        pdf_html = render_template('pdf_template.html', story=story_data, cover_image=cover_image)
        
        pisa.CreatePDF(
            pdf_html, 
            dest=pdf_buffer,
            encoding='utf-8'
        )
        
        pdf_buffer.seek(0)
        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{story_data["title"].replace(" ", "_")}.pdf"'
        
        return response
    except Exception as e:
        return jsonify({
            "error": "Error generating PDF",
            "details": str(e)
        }), 500

# Vercel requires a single route to handle all serverless function calls
app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def catch_all(path):
    return jsonify({'error': 'Invalid endpoint'}), 404

# Development server
if __name__ == '__main__':
    app.run(debug=True)
