import scrapy
import json

from ..items import LandScraperItem

url = "https://basiclandart.com/wp-admin/admin-ajax.php?id=&post_id=0&slug=home&posts_per_page=60&page={}&offset=60&post_type=post&repeater=default&seo_start_page=1&preloaded=false&preloaded_amount=0&order=DESC&orderby=date&action=alm_get_posts&query_type=standard"

class LandSpider(scrapy.Spider):
    name = "land_spider"
    start_urls = [url.format(i) for i in range(15)]
    
    def parse(self, response):
        data = json.loads(response.body)
        selector = scrapy.Selector(text=data['html'], type="html")
        img_links = selector.xpath('//div[@class="posts"]/a/@href').getall()
        yield from response.follow_all(img_links, self.parse_image_page)
    
    def parse_image_page(self, response):
        item = LandScraperItem()
        item['image_urls'] = response.xpath('//figure/img/@src').getall()
        item['tags'] = response.xpath('//div[@class="tags"]/p/a/text()').getall()
        return item
