document.addEventListener("DOMContentLoaded", function() {
    fetch('newsdata_by_category/entertainment/entertainment_newsdata.json')
    .then(response => response.json())
    .then(data => {
        let newsContainer = document.getElementById('news-container');
        
        data.forEach(news => {
            let article = document.createElement('div');
            article.className = 'news-article';
  
            article.innerHTML = `
                <div class="image-container">
                  <img src="${news.image}" alt="News Image" width="200" height="200">
                </div>
                <div class="content-container">
                  <h2 class="headline">${news.headline}</h2>
                  <p class="summary">${news.summary}</p>
                </div>
                <div class="links-container">
                  <a href="${news.url}">link</a>
                </div>
                <div class="source-info">
                  <span>Source:</span> ${news.source}
                </div>
            `;
  
            newsContainer.appendChild(article);
        });
    })
    .catch(error => console.error('Error fetching the news data:', error));
  });
  