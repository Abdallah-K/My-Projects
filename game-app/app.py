from logging import Manager
import requests
import kivymd
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import Screen
from kivy.animation import Animation
from kivymd.uix.imagelist import SmartTileWithLabel
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import ThreeLineAvatarListItem,ImageLeftWidget
from kivymd.toast import toast


Window.size=(300,600)

class Manager(ScreenManager):
    pass


class One(Screen):
    pass


class Two(Screen):
    pass



sm=ScreenManager()
sm.add_widget(One(name = "one"))
sm.add_widget(Two(name = "two"))



class MyApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette="DeepOrange"
        return Manager()
    


    def get_game(self):
        if self.root.ids.two.ids.name.text == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Game-Center",text="Please enter catagory",buttons=[self.close_btn])
            self.dialog.open()
        else:
            anime_api=f"https://www.freetogame.com/api/games?category={self.root.ids.two.ids.name.text}"
            json_data=requests.get(anime_api).json()
            for items in json_data:
                imag=items['thumbnail']
                title=items['title']
                publisher=items['publisher']
                release_date=items['release_date']
                self.li=ThreeLineAvatarListItem(text=title,secondary_text="Publisher: " +str(publisher),tertiary_text="Release_date: "+str(release_date),
                                                theme_text_color="Error")
                img=ImageLeftWidget(source=imag)
                self.li.add_widget(img)
                self.root.ids.two.ids.list.add_widget(self.li)
                toast(f"{self.root.ids.two.ids.name.text} is here!")



    def close_dia(self,obj):
        self.dialog.dismiss()

    def anime(self,widget):
        anim = Animation(pos_hint={"center_y": 1.16})
        anim.start(widget)
    
    def anime1(self,widget):
        anim = Animation(pos_hint={"center_y": .85})
        anim.start(widget)
    

    def icons(self,widget):
        anim=Animation(pos_hint={"center_y":.8})
        anim+=Animation(pos_hint={"center_y":.85})
        anim.start(widget)
    

    def text(self,widget):
        anim=Animation(opacity=0,duration=2)
        anim +=Animation(opacity=1)
        anim.repeat=True
        anim.start(widget)


    def btn(self,widget):
        anim=Animation(opacity=1,duration=3)
        anim.start(widget)
    

    def cen_label(self,widget):
        anim=Animation(pos_hint={"center_x":1.6})
        anim+=Animation(pos_hint={"center_x":0.5})
        anim.start(widget)

    def on_start(self):
        #self.anime(self.root.ids.start.ids.back)
        self.anime1(self.root.ids.one.ids.back1)
        self.icons(self.root.ids.one.ids.icon)
        self.text(self.root.ids.one.ids.label)
        self.btn(self.root.ids.one.ids.btn)
        self.cen_label(self.root.ids.one.ids.txt)
    


    def poster(self):
        if self.root.ids.two.ids.new_name.text == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_d)
            self.dialo=MDDialog(title="Game-Center",text="Please enter catagory",buttons=[self.close_btn])
            self.dialo.open()
        else:
            anime_api=f"https://www.freetogame.com/api/games?category={self.root.ids.two.ids.new_name.text}"
            json_data=requests.get(anime_api).json()
            for items in json_data:
                thu=items['thumbnail']
                title=items['title']
                self.root.ids.two.ids.grid.add_widget(SmartTileWithLabel(source=f"{thu}",text = f"{title}"))
                toast(f"{self.root.ids.two.ids.new_name.text} is here")

    def close_d(self,obj):
        self.dialo.dismiss()




MyApp().run()