import kivymd
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.uix.list import OneLineListItem,ThreeLineAvatarListItem,ThreeLineIconListItem
from kivymd.uix.list import ImageLeftWidget,IconRightWidget,IconLeftWidget
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

Window.size=(300,550)

app_helper="""
Screen:
    MDNavigationLayout:
        ScreenManager:
            Screen:
                MDToolbar:
                    title:"Telegram"
                    pos_hint:{"top":1}
                    left_action_items:[["menu",lambda x:nav_drawer.set_state("open")]]
                    right_action_items:[["account-search",lambda x: app.open_dialog()]]
                    elevation:10
                MDBottomAppBar:
                    MDToolbar:
                        type:"bottom"
                        left_action_items:[["clock"]]
                        mode:"end"
                        icon: 'telegram'
                ScrollView:
                    pos_hint:{"top":0.9}
                    MDList:
                        id: list
        MDNavigationDrawer:
            id:nav_drawer
            BoxLayout:
                orientation:"vertical"
                Image:
                    source:"tele1.jpg"
                    size_hint:(1,0.4)
                MDLabel:
                    text:"    Abdallah Kharoby"
                    size_hint_y:None
                    height:self.texture_size[1]
                MDLabel:
                    text:"    71-982-702"
                    size_hint_y:None
                    height:self.texture_size[1]
                ScrollView:
                    MDList:
                        OneLineIconListItem:
                            text:"New Group"
                            IconLeftWidget:
                                icon:"account-group"
                        OneLineIconListItem:
                            text:"Contacts"
                            IconLeftWidget:
                                icon:"contacts"
                        OneLineIconListItem:
                            text:"Calls"
                            IconLeftWidget:
                                icon:"phone"
                        OneLineIconListItem:
                            text:"People Nearby"
                            IconLeftWidget:
                                icon:"near-me"
                        OneLineIconListItem:
                            text:"Saved Messages"
                            IconLeftWidget:
                                icon:"message-bookmark"
                        OneLineIconListItem:
                            text:"settings"
                            IconLeftWidget:
                                icon:"account-settings"
                        OneLineIconListItem:
                            text:"Invite Friends"
                            IconLeftWidget:
                                icon:"account-plus"
                        OneLineIconListItem:
                            text:"Telegram Features"
                            IconLeftWidget:
                                icon:"telegram"
"""






class MyApp(MDApp):
    def build(self):
        self.title="Telegram"
        #self.theme_cls.primary_palette="Red"
        screen=Builder.load_string(app_helper)
        
        return screen
    
    def on_start(self):
        for i in range(15):
            #image=ImageLeftWidget(source="logo.png")
            icon=IconLeftWidget(icon="face-profile-woman")
            list_view=ThreeLineIconListItem(text="Abdallah",secondary_text="Hello world",
                                            tertiary_text="Online")
            list_view.add_widget(icon)
            self.root.ids.list.add_widget(list_view)
        
    
    def open_dialog(self):
        close_btn=MDFlatButton(text="close",on_release=self.close_dia)
        self.dialog=MDDialog(title="Telegram clone",text="Have a look on Telegrame UI",
                            buttons=[close_btn])
        self.dialog.open()
    

    def close_dia(self,obj):
        self.dialog.dismiss()




MyApp().run()





