# -*- coding: utf-8 -*-

# Scrapy settings for land_scraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'land_scraper'

SPIDER_MODULES = ['land_scraper.spiders']
NEWSPIDER_MODULE = 'land_scraper.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'land_scraper (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {'scrapy.pipelines.images.ImagesPipeline':1}

IMAGES_STORE = '../land_imgs'